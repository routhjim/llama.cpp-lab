#pragma once

#include <algorithm>
#include <cstdlib>

// Adaptive draft depth controller for MTP speculative decoding (draft-mtp-adaptive).
//
// Hysteresis state machine with a climb counter and a weighted drop-pressure
// accumulator. The depth N climbs one step after N_CLIMB(N) consecutive verifies
// that accepted every drafted token. The climb cost is low at the floor and at
// depth, high in the middle: 2 at depth 1, 4 at depth 2, 10 at depth 3, then
// 6/3/2/2 from depth 4 upward. Getting from the floor to depth 3 needs only 6
// full accepts, but pushing past 3 (where prose acceptance collapses) costs 10
// full accepts of 3-token drafts, which predictable content clears quickly and
// marginal content never does. Any miss adds (n_draft - n_accepted) to a
// drop-pressure accumulator; when it reaches depth * 5 the depth drops one step
// and the pressure resets. A near miss (n_draft-1) adds 1, a total miss adds
// n_draft, so high depths fall quickly while low depths hold. The drop budget
// scales with depth but never drops below 20, so shallow depths shed bad content
// quickly without collapsing to the floor on a few bad rounds; deep depths hold
// a little longer. At the floor no pressure accumulates at all. The depth starts
// at the floor max(1, --spec-draft-n-min-adaptive) and stays in
// [floor, n_max]; --spec-draft-n-max bounds the upper end of the adaptive
// range.
struct common_speculative_adaptive {
    int n_cur   = 0; // current adaptive draft depth N
    int n_climb = 0; // consecutive verifies that accepted every drafted token
    int n_drop  = 0; // accumulated drop pressure: sum of (n_draft - n_accepted)

    // consecutive full accepts needed to climb one step from depth N; low at the
    // floor and at depth, high in the middle where acceptance is marginal
    static int climb_threshold(int depth) {
        switch (depth) {
            case 1: return 2;
            case 2: return 4;
            case 3: return 10; // hardened 3->4 barrier: keeps prose/reasoning pinned
            case 4: return 6;
            case 5: return 3;
            case 6: return 2;
            default: return 2; // depth >= 7
        }
    }

    // accumulated (n_draft - n_accepted) needed to drop one step from depth N;
    // scaled by depth, with a floor so shallow depths do not collapse too fast
    static int drop_pressure(int depth) {
        return std::max(depth * 5, 20);
    }

    // ---- ROI mode (LLAMA_ADAPTIVE_ROI=1) ---------------------------------------------
    // The streak/pressure machine above assumes the verify step costs the same at every
    // depth, which holds on dense models. On a MoE at np>1 every draft position widens
    // the verify batch and the union of routed experts, so a position only pays when the
    // tokens it adds outweigh the step cost it adds. This mode tracks the unconditional
    // acceptance rate of each draft position as an EMA and picks the depth maximising
    // expected tokens per step cost, with step cost modelled as 1 + r*depth
    // (LLAMA_ADAPTIVE_COST_RATIO, default 0.33 = Flash-Next XL at np=4 measured 1->2).
    // A position that is not being drafted cannot be observed, so the controller probes
    // depth+1 for ROI_PROBE_LEN steps every ROI_PROBE_PERIOD steps and keeps it only if
    // it measured better. Dropping needs no probe: depth-1's positions are observed at
    // depth. EMAs survive reset() (a slot's conversation stays alike); depth restarts at
    // the best warm depth.
    static constexpr int   ROI_MAX          = 16;
    static constexpr float ROI_ALPHA        = 1.0f / 16.0f;
    static constexpr int   ROI_WARMUP       = 16;
    static constexpr int   ROI_PROBE_PERIOD = 96;
    static constexpr int   ROI_PROBE_LEN    = 24;
    static constexpr float ROI_HYST         = 0.05f;

    float p_acc[ROI_MAX + 1] = {}; // EMA of P(draft position k accepted), k = 1..ROI_MAX
    int   n_obs[ROI_MAX + 1] = {}; // observations of position k
    int   steps_at_depth = 0;
    int   probe_left     = 0;      // >0 while probing depth probe_base+1
    int   probe_base     = 0;

    static bool roi_mode() {
        static const bool v = [] { const char * e = getenv("LLAMA_ADAPTIVE_ROI"); return e && atoi(e) != 0; }();
        return v;
    }
    static float cost_ratio() {
        static const float r = [] { const char * e = getenv("LLAMA_ADAPTIVE_COST_RATIO"); return e ? (float) atof(e) : 0.33f; }();
        return r;
    }

    // expected accepted draft tokens per step at depth d (positions are a prefix, so
    // the unconditional rates already carry the dependency on earlier positions)
    float roi_expected(int d) const {
        float e = 0.0f;
        for (int k = 1; k <= std::min(d, ROI_MAX); ++k) { e += p_acc[k]; }
        return e;
    }
    // tokens per unit step cost at depth d, counting the target's own token
    float roi_rate(int d) const {
        return (1.0f + roi_expected(d)) / (1.0f + cost_ratio() * (float) d);
    }
    bool roi_warm(int d) const {
        return d >= 1 && d <= ROI_MAX && n_obs[d] >= ROI_WARMUP;
    }

    void roi_reset(int cap, int floor) {
        int best = floor;
        for (int d = floor + 1; d <= cap && d <= ROI_MAX; ++d) {
            if (roi_warm(d) && roi_rate(d) > roi_rate(best) * (1.0f + ROI_HYST)) { best = d; }
        }
        n_cur          = std::min(best, cap);
        steps_at_depth = 0;
        probe_left     = 0;
    }

    void roi_update(int n_draft, int n_accepted, int cap, int floor) {
        for (int k = 1; k <= std::min(n_draft, ROI_MAX); ++k) {
            const float x = n_accepted >= k ? 1.0f : 0.0f;
            p_acc[k] += ROI_ALPHA * (x - p_acc[k]);
            n_obs[k]++;
        }
        steps_at_depth++;

        if (probe_left > 0) {
            if (--probe_left == 0) {
                // keep the probed depth only if it measured better than where we came from
                if (!(roi_rate(n_cur) > roi_rate(probe_base) * (1.0f + ROI_HYST))) {
                    n_cur = probe_base;
                }
                steps_at_depth = 0;
            }
            return;
        }

        // drop: one shallower is observable right now and pays better
        if (n_cur > floor && roi_warm(n_cur) && roi_rate(n_cur - 1) > roi_rate(n_cur) * (1.0f + ROI_HYST)) {
            n_cur--;
            steps_at_depth = 0;
            return;
        }

        // climb: only by probing, since position n_cur+1 is unobserved
        if (n_cur < cap && steps_at_depth >= ROI_PROBE_PERIOD) {
            probe_base     = n_cur;
            n_cur++;
            probe_left     = ROI_PROBE_LEN;
            steps_at_depth = 0;
        }
    }
    // ---------------------------------------------------------------------------------

    // reset to the floor max(1, n_min_adaptive), bounded by the ceiling n_max;
    // the controller climbs from there once acceptance feedback arrives
    void reset(int n_max, int n_min_adaptive) {
        const int cap   = std::max(1, n_max);
        const int floor = std::max(1, n_min_adaptive);

        if (roi_mode()) {
            roi_reset(cap, floor);
            return;
        }

        n_cur   = std::min(floor, cap);
        n_climb = 0;
        n_drop  = 0;
    }

    // feed one verification result: n_draft is the number of tokens this
    // implementation drafted, n_accepted the number the target accepted
    void update(int n_draft, int n_accepted, int n_max, int n_min_adaptive) {
        if (n_draft <= 0) {
            return;
        }

        const int cap   = std::max(1, n_max);
        const int floor = std::max(1, n_min_adaptive);

        if (roi_mode()) {
            roi_update(n_draft, n_accepted, cap, floor);
            return;
        }

        if (n_accepted == n_draft) {
            n_drop = 0;

            // full acceptance: reset the drop pressure, accumulate the climb streak
            if (n_cur < cap && ++n_climb >= climb_threshold(n_cur)) {
                n_cur++;
                n_climb = 0;
            }
        } else {
            n_climb = 0;

            // any miss adds (n_draft - n_accepted) to the drop pressure; drop one
            // step when the accumulated pressure reaches the depth-scaled budget
            if (n_cur > floor) {
                n_drop += n_draft - n_accepted;
                if (n_drop >= drop_pressure(n_cur)) {
                    n_cur--;
                    n_drop = 0;
                }
            }
        }
    }
};
