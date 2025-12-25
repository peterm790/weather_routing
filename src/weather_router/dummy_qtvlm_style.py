"""
Minimal isochrone router with Newton refinement (schematic).

Purpose / intent
- This is a compact, heavily-commented Python "schematic" of the C++ ROUTAGE
  algorithm you showed earlier. It focuses on the algorithmic structure:
    * isochrone expansion (grow fronts in time),
    * a cheap per-heading estimate (speed at origin * dt),
    * an optional Newton refinement that runs a forward simulator to solve
      travel_time(distance) == timestep,
    * basic pruning (bucketing + wake),
    * backtracking to build the final route.
- It intentionally omits many production features from the C++ code:
  intersection-heavy geometry, shape-based pruning, multithreading,
  fine-grained smoothing, GUI/Projection, full tack modeling, and performance
  optimizations. Treat this as a readable blueprint, not a full replacement.

Key conventions and units
- Lat/lon are (latitude, longitude) in decimal degrees.
- Polar.get_speed(tws, twa_abs) must return speed in knots.
- Distances are in nautical miles (nm); geopy.great_circle(...).nm is used.
- Time is represented in seconds where needed; step_hours is the time-step in hours.
- get_wind(t, lat, lon) must return (twd_deg, tws_knots).
- Optional get_current(t, lat, lon) may be provided; otherwise currents are ignored.

How to use
- Provide implementations for:
    * polar.get_speed(tws, twa_abs)
    * get_wind(t, lat, lon)
    * (optional) get_current(t, lat, lon)
    * (optional) point_validity(lat, lon)
- Configure time_steps and step_hours to match your weather data cadence.
- Set use_refine=True to enable Newton-refinement (more accurate, much slower).

Structure
- Node: minimal state for each isochrone point (lat, lon, eta, parent, heading).
- calculate_time_route: forward simulator used by Newton and arrival checks.
- newton_refine_distance: root-finds distance along a heading so forward-sim time == timestep.
- expand_isochrone: produces candidate children for a frontier (cheap -> optional refine).
- route: main loop that grows isochrones until the goal is reached (or time runs out).
"""

from dataclasses import dataclass
from typing import Optional, Callable, List, Tuple
import math
import numpy as np
import geopy.distance

# --------------------------------------
# Utility functions (small geodesy helpers)
# --------------------------------------

def bearing_between(a: Tuple[float,float], b: Tuple[float,float]) -> float:
    """Return initial bearing (degrees) from point a(lat,lon) to b(lat,lon)."""
    lat1 = math.radians(a[0]); lat2 = math.radians(b[0])
    dlon = math.radians(b[1]-a[1])
    x = math.sin(dlon) * math.cos(lat2)
    y = math.cos(lat1)*math.sin(lat2) - math.sin(lat1)*math.cos(lat2)*math.cos(dlon)
    # atan2 gives bearing in radians; convert to degrees and normalize 0..360
    return (math.degrees(math.atan2(x,y)) + 360.0) % 360.0

def normalize_angle180(a: float) -> float:
    """Normalize angle to [-180, 180). Useful for TWA computations."""
    return (a + 180) % 360 - 180

def great_circle_distance_nm(a: Tuple[float,float], b: Tuple[float,float]) -> float:
    """Great-circle distance in nautical miles between two (lat, lon) points."""
    return geopy.distance.great_circle(a, b).nm

# --------------------------------------
# Core data container
# --------------------------------------

@dataclass
class Node:
    """
    Minimal state for a point on an isochrone.

    - lat, lon: geographic coordinates in degrees
    - eta: epoch seconds for arrival time at this node
    - parent: Node from previous isochrone that generated this node
    - cap_origin: heading (deg) used from the parent to this node
    - wind: optional cached wind at the origin (twd_deg, tws_knots)
    - dist_from_origin: distance (nm) travelled from parent to this node
    """
    lat: float
    lon: float
    eta: float
    parent: Optional['Node']
    cap_origin: float
    wind: Optional[Tuple[float,float]] = None
    dist_from_origin: float = 0.0

# --------------------------------------
# Router core (minimal)
# --------------------------------------

class MinimalIsochronalRouter:
    """
    Minimal implementation of isochrone routing with optional Newton refinement.

    Parameters of interest:
    - polar: object with get_speed(tws, twa_abs) -> knots
    - get_wind(t, lat, lon): function returning (twd_deg, tws_knots)
    - time_steps: sequence of times (numpy.datetime64 or epoch seconds)
    - step_hours: hours per isochrone step
    - start_point / end_point: (lat, lon)
    - use_refine: whether to run Newton refinement per candidate (expensive)
    """
    def __init__(
        self,
        polar,
        get_wind: Callable,
        time_steps: List,
        step_hours: float,
        start_point: Tuple[float,float],
        end_point: Tuple[float,float],
        headings_spread: int = 110,
        heading_step_deg: float = 10.0,
        use_refine: bool = True,
        vac_len_sec: int = 600,
        point_validity: Optional[Callable[[float,float], bool]] = None,
        get_current: Optional[Callable] = None,
        wake_lim_deg: float = 45.0,
        bucket_rounding_deg: float = 3,
    ):
        # core dependencies
        self.polar = polar
        self.get_wind = get_wind
        self.get_current = get_current  # optional current provider

        # time and step logic
        self.time_steps = time_steps
        self.step_hours = step_hours
        self.tstep_sec = step_hours * 3600.0

        # geometry and search params
        self.start_point = start_point
        self.end_point = end_point
        self.spread = headings_spread
        self.step_heading = heading_step_deg
        self.use_refine = use_refine
        self.vac_len_sec = vac_len_sec  # integrator sub-step in seconds

        # simple validators and pruning params
        self.point_validity = (point_validity if point_validity is not None else (lambda lat,lon: True))
        self.wake_lim = wake_lim_deg
        self.bucket_round = bucket_rounding_deg

        # output
        self.isochrones: List[List[Node]] = []

    # --------------------------------------
    # Forward simulator (used for Newton refinement & exact arrival checks)
    # --------------------------------------
    def calculate_time_route(self, origin: Node, target_lat: float, target_lon: float, eta0: float, max_time_sec: float = None) -> float:
        """
        Forward-integrate motion from origin toward target point and return travel time in seconds.

        - origin: Node (starting position & time)
        - target_lat/lon: destination point to measure time to
        - eta0: starting epoch seconds
        - max_time_sec: optional cap to abort the integration early

        Simple integrator behaviour:
        - at each small step (vac_len_sec):
            * sample local wind (and optionally current)
            * compute instantaneous heading (bearing to target) and TWA
            * get boat speed from polar (knots)
            * apply current via vector sum if provided
            * move a short distance along this heading and update t/remaining distance
        - returns a large sentinel (1e9) on failure or if exceed max_time_sec
        """
        max_time_sec = max_time_sec if max_time_sec is not None else self.tstep_sec * 10.0

        lat = origin.lat
        lon = origin.lon
        t = eta0
        remaining = great_circle_distance_nm((lat,lon), (target_lat, target_lon))
        if remaining == 0.0:
            return 0.0

        elapsed = 0.0
        # safety limit for iterations
        iter_limit = int(max_time_sec / max(1.0, self.vac_len_sec)) + 5

        for _ in range(iter_limit):
            # sample wind at current position/time (must be provided by user)
            try:
                twd, tws = self.get_wind(t, lat, lon)
            except Exception:
                # missing weather => cannot simulate reliably
                return 1e9

            # heading towards target (we pick the great-circle initial bearing)
            cap = bearing_between((lat,lon), (target_lat, target_lon))
            # true wind angle relative to boat (TWA) in [-180,180)
            twa = normalize_angle180(cap - twd)

            # base boat speed from polar (knots) based on local tws and absolute TWA
            speed = self.polar.get_speed(tws, abs(twa))

            # if current provider available, combine vectors (very simple vector sum)
            if self.get_current:
                try:
                    cdir, cspeed = self.get_current(t, lat, lon)
                    # convert polar vectors into x/y components (nautical miles per hour)
                    bx = speed * math.cos(math.radians(cap))
                    by = speed * math.sin(math.radians(cap))
                    cx = cspeed * math.cos(math.radians(cdir))
                    cy = cspeed * math.sin(math.radians(cdir))
                    sog_x = bx + cx
                    sog_y = by + cy
                    sog = math.hypot(sog_x, sog_y)
                except Exception:
                    sog = speed
            else:
                sog = speed

            # distance in this small integration step (nm)
            move_nm = sog * (self.vac_len_sec / 3600.0)

            # If this step reaches or overshoots the target, compute partial time to finish
            if move_nm >= remaining:
                # avoid division by zero; convert hours -> seconds
                if sog > 0:
                    elapsed += remaining / sog * 3600.0
                else:
                    elapsed += 1e9
                return elapsed

            # Advance location along the current heading by move_nm
            # (we use geopy to compute a small-step destination)
            dest = geopy.distance.great_circle(nautical=move_nm).destination((lat,lon), cap)
            lat, lon = dest.latitude, dest.longitude

            # time & loop bookkeeping
            t += self.vac_len_sec
            elapsed += self.vac_len_sec
            remaining = great_circle_distance_nm((lat,lon), (target_lat, target_lon))

            # safety guard: abort if taking too long
            if elapsed > max_time_sec:
                return 1e9

        # if we exit the loop without arrival, consider it unreachable in allowed time
        return 1e9

    # --------------------------------------
    # Newton refinement (solve travel_time(x) == timestep)
    # --------------------------------------
    def route_time_function(self, origin: Node, cap_deg: float, x_nm: float, eta0: float) -> float:
        """
        Convenience wrapper: get travel time to a point x_nm along cap_deg from origin.
        Returns time in seconds (or large sentinel on failure).
        """
        dest = geopy.distance.great_circle(nautical=x_nm).destination((origin.lat, origin.lon), cap_deg)
        lat_t, lon_t = dest.latitude, dest.longitude
        return self.calculate_time_route(origin, lat_t, lon_t, eta0, max_time_sec=self.tstep_sec * 10.0)

    def newton_refine_distance(self, origin: Node, cap_deg: float, x0_nm: float, eta0: float, tol_sec: float = 1.0, max_iter: int = 20) -> Optional[float]:
        """
        Abstract:
        - Solve g(x) = route_time(x) - timestep == 0 for x (distance in nm) using Newton-like method.
        - route_time(x) is expensive (forward simulation). We compute finite-difference derivative.
        - If derivative is unstable or near-zero, fall back to a bracketed bisection search.

        Returns:
        - x (nm) if converged within tol_sec
        - None if no reliable solution found
        """
        target = self.tstep_sec
        x = max(0.0, x0_nm)

        for it in range(max_iter):
            y = self.route_time_function(origin, cap_deg, x, eta0) - target
            if abs(y) <= tol_sec:
                # converged: travel_time within tolerance of desired step
                return x

            # finite-difference derivative estimation (centered)
            dx = max(1e-3, x * 0.01)  # minimal perturbation in nm
            y_r = self.route_time_function(origin, cap_deg, x + dx, eta0) - target
            y_l = self.route_time_function(origin, cap_deg, max(0.0, x - dx), eta0) - target
            denom = (y_r - y_l)

            # If derivative is near-zero or noisy, fallback to bracketed bisection
            if abs(denom) < 1e-6:
                # try to bracket a root between [0, upper]
                upper = max(x + 1.0, x * 2.0 + 1.0)
                lo, hi = 0.0, upper
                vlo = self.route_time_function(origin, cap_deg, lo, eta0) - target
                vhi = self.route_time_function(origin, cap_deg, hi, eta0) - target
                if vlo * vhi > 0:
                    # both same sign, unable to bracket => give up
                    return None
                # bisection loop (simple and robust)
                for _ in range(30):
                    mid = 0.5 * (lo + hi)
                    vmid = self.route_time_function(origin, cap_deg, mid, eta0) - target
                    if abs(vmid) <= tol_sec:
                        return mid
                    if vlo * vmid <= 0:
                        hi = mid
                        vhi = vmid
                    else:
                        lo = mid
                        vlo = vmid
                return None

            deriv = denom / (2.0 * dx)
            x_new = x - y / deriv
            # guard against negative or huge update
            if x_new < 0.0:
                x_new = x / 2.0
            # damping the update to stabilize Newton iterations
            x = max(0.0, 0.5 * x + 0.5 * x_new)

        # did not converge inside max_iter
        return None

    # --------------------------------------
    # Expand a single frontier (one time-step) -> produce candidate children
    # --------------------------------------
    def expand_isochrone(self, frontier: List[Node], t_step) -> List[Node]:
        """
        For every node in the frontier:
        - sample headings around the bearing-to-goal (spread / step_heading)
        - compute cheap estimate: base_speed_at_origin * dt
        - optionally refine that distance with Newton refinement (very expensive)
        - validate the candidate (point_validity) and collect
        After all children are collected:
        - do coarse bucket pruning (one point per lat/lon bucket keeping the closest to goal)
        - simple wake pruning: discard points in wake cones of better points
        """
        new_nodes: List[Node] = []

        for node in frontier:
            # compute search sector centered on bearing-to-goal
            brg_to_goal = bearing_between((node.lat, node.lon), self.end_point)
            lower = int(brg_to_goal - self.spread)
            upper = int(brg_to_goal + self.spread) + 1

            # sample headings in the sector
            for h in range(lower, upper, int(self.step_heading)):
                cap = (h + 360) % 360

                # cheap weather sample at origin (fast, but not physically exact along the leg)
                try:
                    twd, tws = self.get_wind(t_step, node.lat, node.lon)
                except Exception:
                    # if wind sample fails for any reason, skip this heading
                    continue

                twa = normalize_angle180(cap - twd)
                base_speed = self.polar.get_speed(tws, abs(twa))  # knots

                # naive distance estimate (cheap): distance = speed * hours
                est_dist = base_speed * (self.step_hours)

                final_dist = est_dist
                if self.use_refine:
                    # attempt Newton refinement: find distance that results in exactly one timestep
                    refined = self.newton_refine_distance(node, cap, est_dist, node.eta)
                    if refined is not None:
                        final_dist = refined
                    else:
                        # fallback: keep cheap estimate (could also skip candidate)
                        final_dist = est_dist

                # compute resulting lat/lon from origin along cap by final_dist
                dest = geopy.distance.great_circle(nautical=final_dist).destination((node.lat, node.lon), cap)
                lat2, lon2 = dest.latitude, dest.longitude

                # basic point validity check (land mask, etc.)
                if not self.point_validity(lat2, lon2):
                    continue

                # construct the child node
                child = Node(
                    lat=lat2,
                    lon=lon2,
                    eta=node.eta + self.tstep_sec,
                    parent=node,
                    cap_origin=cap,
                    wind=(twd, tws),
                    dist_from_origin=final_dist
                )
                new_nodes.append(child)

        # If no candidates, return empty list
        if not new_nodes:
            return []

        # -----------------------------
        # Bucket pruning (coarse)
        # Keep a single representative per rounded (lat,lon) bucket.
        # Representative chosen as point closest to goal.
        # -----------------------------
        buckets = {}
        for n in new_nodes:
            key = (round(n.lat, int(self.bucket_round)), round(n.lon, int(self.bucket_round)))
            dist_goal = great_circle_distance_nm((n.lat, n.lon), self.end_point)
            if key not in buckets or dist_goal < buckets[key][0]:
                buckets[key] = (dist_goal, n)
        pruned = [v[1] for v in buckets.values()]

        # -----------------------------
        # Wake-style pruning (simple)
        # For each candidate n, check whether there exists a "better" candidate m (closer to goal)
        # whose backbearing cone contains the bearing from m to n. If so, n is shadowed.
        # -----------------------------
        survivors = []
        for i, n in enumerate(pruned):
            keep = True
            for j, m in enumerate(pruned):
                if i == j:
                    continue
                backbearing = (bearing_between((m.lat, m.lon), self.end_point) + 180.0) % 360.0
                bearing_mn = bearing_between((m.lat, m.lon), (n.lat, n.lon))
                diff = normalize_angle180(bearing_mn - backbearing)
                # if m is closer to goal and n lies within m's "wake" cone, drop n
                if abs(diff) <= self.wake_lim and great_circle_distance_nm((m.lat, m.lon), self.end_point) < great_circle_distance_nm((n.lat, n.lon), self.end_point):
                    keep = False
                    break
            if keep:
                survivors.append(n)

        return survivors

    # --------------------------------------
    # Main isochrone loop
    # --------------------------------------
    def route(self):
        """
        Grow isochrones for each time_step until the finish is reached (or we run out of time).
        - returns a backtracked list of (lat, lon, eta) points representing the route.
        """
        # initial node (start). Convert first time-step element to epoch seconds if numpy.datetime64
        try:
            # numpy.datetime64 -> seconds since epoch
            t0 = int(self.time_steps[0].astype('datetime64[s]').astype(int))
        except Exception:
            # assume already epoch seconds
            t0 = self.time_steps[0]

        start_node = Node(lat=self.start_point[0], lon=self.start_point[1], eta=t0, parent=None, cap_origin=0.0)
        self.isochrones = []
        frontier = [start_node]

        # convert all time steps to epoch seconds (if possible)
        try:
            time_secs = [int(t.astype('datetime64[s]').astype(int)) for t in self.time_steps]
        except Exception:
            time_secs = self.time_steps

        for idx, t in enumerate(time_secs):
            # expand frontier for this step
            new_frontier = self.expand_isochrone(frontier, t)
            if not new_frontier:
                print("no more candidates at step", idx)
                break

            # store isochrone (list of Node)
            self.isochrones.append(new_frontier)

            # check arrival: find minimum distance to goal in this isochrone
            dists = [great_circle_distance_nm((n.lat, n.lon), self.end_point) for n in new_frontier]
            min_dist = min(dists)
            print(f"step {idx}: nodes {len(new_frontier)}, min_dist_to_goal={min_dist:.2f} nm")

            # arrival threshold (simple). In real system you might compute exact time-to-arrival.
            if min_dist <= 1.0:
                # pick best node (closest) and backtrack route
                best = new_frontier[int(np.argmin(dists))]
                return self.backtrack(best)

            frontier = new_frontier

        # if we exhausted time steps without reaching finish, return best available path
        if self.isochrones:
            last = self.isochrones[-1]
            dists = [great_circle_distance_nm((n.lat, n.lon), self.end_point) for n in last]
            best = last[int(np.argmin(dists))]
            return self.backtrack(best)

        # no candidates at all
        return []

    # --------------------------------------
    # Backtrack to build final route list
    # --------------------------------------
    def backtrack(self, node: Node) -> List[Tuple[float,float,float]]:
        """
        Follow parent pointers from node back to the start, producing a sequence
        of (lat, lon, eta) from start -> finish.
        """
        path = []
        cur = node
        while cur is not None:
            path.append((cur.lat, cur.lon, cur.eta))
            cur = cur.parent
        path.reverse()
        return path

# --------------------------------------
# Example usage / placeholders (fill with real implementations)
# --------------------------------------
if __name__ == "__main__":
    # Minimal toy polar implementation for testing.
    class DummyPolar:
        def get_speed(self, tws, twa_abs):
            # crude example: maximum ~6 kn at best angle, linear drop to 0 at 180 deg
            return max(0.5, 6.0 * (1.0 - abs(twa_abs) / 180.0))

    # Constant wind from west at 12 kn
    def dummy_get_wind(t, lat, lon):
        return (270.0, 12.0)

    # Everything is valid (no landmask)
    def dummy_valid(lat, lon):
        return True

    # build hourly time steps for 24 hours (numpy.datetime64)
    now = np.datetime64('2025-12-24T00:00')
    time_steps = np.array([now + np.timedelta64(h, 'h') for h in range(0, 24)])

    # create router with Newton refinement OFF by default (expensive)
    router = MinimalIsochronalRouter(
        polar=DummyPolar(),
        get_wind=dummy_get_wind,
        time_steps=time_steps,
        step_hours=1.0,
        start_point=(48.0, -4.0),
        end_point=(43.0, -1.0),
        use_refine=False,    # set True to enable Newton refinement (slower, more accurate)
        point_validity=dummy_valid
    )

    path = router.route()
    print("path:", path)