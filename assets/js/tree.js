/*
  The growing tree that frames the landing page.

  A band of scattered points is connected into a tree, and the tree is animated
  as it grows. The settled settings from the demo page (/tree-demo/, kept
  around for tweaking) are the constants below.

  The band is a ring around the content, seeded at top centre, so growth runs
  away in both directions and the two halves meet at the bottom.
*/
(() => {
  const SVGNS = "http://www.w3.org/2000/svg";

  const SETTINGS = {
    band: 58, // ring thickness, in svg units, where the frame is 1000 wide
    spacing: 9, // minimum distance between scattered points
    turn: 2, // how much a change of direction costs when growing
    slowdown: 0.25, // speed of each level of branch relative to the one above
    catchup: 3.5, // how much time accelerates once the skeleton is drawn
    speed: 130, // svg units per second
    ramp: 0.8, // seconds over which catch-up eases in
    resettle: 800, // ms of stillness after a resize before the tree regrows
    minChange: 24, // px of size change worth regenerating for
  };

  const mulberry32 = (seed) =>
    function () {
      seed |= 0;
      seed = (seed + 0x6d2b79f5) | 0;
      let t = Math.imul(seed ^ (seed >>> 15), 1 | seed);
      t = (t + Math.imul(t ^ (t >>> 7), 61 | t)) ^ t;
      return ((t ^ (t >>> 14)) >>> 0) / 4294967296;
    };

  // Blue-noise scatter, masked to the ring: uniform random clumps and leaves
  // bald patches, which makes the tree look accidental.
  function poissonDisc(shape, radius, rand, k = 25) {
    const cell = radius / Math.SQRT2;
    const cols = Math.ceil(shape.w / cell);
    const rows = Math.ceil(shape.h / cell);
    const grid = new Array(cols * rows).fill(-1);
    const points = [];
    const active = [];
    const gi = (x, y) => Math.floor(y / cell) * cols + Math.floor(x / cell);

    const fits = (x, y) => {
      if (x < 0 || y < 0 || x >= shape.w || y >= shape.h) return false;
      if (!shape.inside(x, y)) return false;
      const gx = Math.floor(x / cell);
      const gy = Math.floor(y / cell);
      for (let iy = Math.max(0, gy - 2); iy <= Math.min(rows - 1, gy + 2); iy++) {
        for (let ix = Math.max(0, gx - 2); ix <= Math.min(cols - 1, gx + 2); ix++) {
          const idx = grid[iy * cols + ix];
          if (idx === -1) continue;
          const p = points[idx];
          if ((p[0] - x) ** 2 + (p[1] - y) ** 2 < radius * radius) return false;
        }
      }
      return true;
    };

    const first = [shape.w / 2, Math.min(SETTINGS.band / 2, shape.h / 2)]; // top centre
    points.push(first);
    grid[gi(first[0], first[1])] = 0;
    active.push(0);

    while (active.length) {
      const pick = Math.floor(rand() * active.length);
      const from = points[active[pick]];
      let placed = false;
      for (let i = 0; i < k; i++) {
        const angle = rand() * Math.PI * 2;
        const dist = radius * (1 + rand());
        const x = from[0] + Math.cos(angle) * dist;
        const y = from[1] + Math.sin(angle) * dist;
        if (!fits(x, y)) continue;
        points.push([x, y]);
        grid[gi(x, y)] = points.length - 1;
        active.push(points.length - 1);
        placed = true;
        break;
      }
      if (!placed) active.splice(pick, 1);
    }
    return points;
  }

  // Greedy growth in Prim's style, with a turn penalty so the tree favours
  // continuing straight over doubling back.
  function grow(points, shape) {
    const n = points.length;
    const seed = [shape.w / 2, Math.min(SETTINGS.band / 2, shape.h / 2)];

    let root = 0;
    let bestRoot = Infinity;
    for (let i = 0; i < n; i++) {
      const d = Math.hypot(points[i][0] - seed[0], points[i][1] - seed[1]);
      if (d < bestRoot) {
        bestRoot = d;
        root = i;
      }
    }

    const inTree = new Array(n).fill(false);
    const cost = new Float64Array(n).fill(Infinity);
    const parent = new Int32Array(n).fill(-1);
    const dir = points.map(() => null);
    const children = points.map(() => []);
    cost[root] = 0;

    const weight = (u, v) => {
      const dx = points[v][0] - points[u][0];
      const dy = points[v][1] - points[u][1];
      const len = Math.hypot(dx, dy) || 1e-6;
      let w = len;

      // The ring is a loop and a tree cannot hold a loop, so it has to break
      // somewhere. Left alone that happens at an arbitrary edge, sending
      // growth most of the way round one side. Making the cut prohibitively
      // expensive at the bottom centre -- opposite the seed -- splits the ring
      // into halves that meet at the bottom. Expensive rather than removed, so
      // the graph stays connected whatever the scatter looks like.
      if (shape.cut(points[u], points[v])) w *= 1000;

      if (dir[u]) {
        const cos = (dx * dir[u][0] + dy * dir[u][1]) / len;
        w *= 1 + SETTINGS.turn * (1 - cos) * 0.5;
      }
      return w;
    };

    for (let iter = 0; iter < n; iter++) {
      let u = -1;
      let best = Infinity;
      for (let v = 0; v < n; v++) {
        if (!inTree[v] && cost[v] < best) {
          best = cost[v];
          u = v;
        }
      }
      if (u === -1) break;
      inTree[u] = true;
      if (parent[u] !== -1) {
        children[parent[u]].push(u);
        const dx = points[u][0] - points[parent[u]][0];
        const dy = points[u][1] - points[parent[u]][1];
        const len = Math.hypot(dx, dy) || 1;
        dir[u] = [dx / len, dy / len];
      }
      for (let v = 0; v < n; v++) {
        if (inTree[v]) continue;
        const w = weight(u, v);
        if (w < cost[v]) {
          cost[v] = w;
          parent[v] = u;
        }
      }
    }

    return { points, children, root };
  }

  // How far the tree reaches below each node: the length of the longest
  // downward path from it. This is the measure of how "main" a branch is.
  // Leaf count is the obvious alternative and it is wrong: where a line
  // continues one way and a bushy cluster of twigs sits the other, leaf count
  // picks the cluster and demotes the continuing line, so one side of the ring
  // ends up several levels down and crawling.
  function reachLengths({ points, children, root }) {
    const reach = new Float64Array(points.length);
    const order = [];
    const stack = [root];
    while (stack.length) {
      const n = stack.pop();
      order.push(n);
      for (const c of children[n]) stack.push(c);
    }
    for (let i = order.length - 1; i >= 0; i--) {
      const n = order[i];
      for (const c of children[n]) {
        const len = Math.hypot(points[n][0] - points[c][0], points[n][1] - points[c][1]);
        reach[n] = Math.max(reach[n], len + reach[c]);
      }
    }
    return reach;
  }

  // Split the tree into runs: from the root, follow the child the tree reaches
  // furthest through, and that run is the skeleton. Branches off it are one
  // level down, and so on, recursively. The root is a special case -- all of
  // its children stay at level 0, so both directions around the ring count as
  // skeleton rather than one side being demoted.
  function skeletonRuns(tree) {
    const { points, children, root } = tree;
    const reach = reachLengths(tree);
    const through = (from, c) =>
      Math.hypot(points[from][0] - points[c][0], points[from][1] - points[c][1]) + reach[c];
    const main = (from, kids) => kids.reduce((a, b) => (through(from, b) > through(from, a) ? b : a));

    const runs = [];
    const queue = [{ from: root, child: null, level: 0 }];

    while (queue.length) {
      const item = queue.shift();
      const seq = [item.from];
      let cur = item.from;
      let next = item.child;

      if (next === null) {
        if (!children[cur].length) continue;
        next = main(cur, children[cur]);
        for (const c of children[cur]) if (c !== next) queue.push({ from: cur, child: c, level: item.level });
      }

      while (next !== null) {
        seq.push(next);
        cur = next;
        const kids = children[cur];
        if (!kids.length) break;
        const h = main(cur, kids);
        for (const c of kids) if (c !== h) queue.push({ from: cur, child: c, level: item.level + 1 });
        next = h;
      }

      runs.push({ seq, level: item.level });
    }
    return runs;
  }

  // Everything starts as soon as it is connected -- no phases -- but each
  // level advances more slowly, so the skeleton outruns its own detail. The
  // falloff is harmonic rather than exponential: compounding would leave deep
  // twigs crawling for an age after everything else has finished.
  function schedule(tree, runs) {
    const k = 1 / SETTINGS.slowdown - 1;
    const len = (a, b) => Math.hypot(tree.points[a][0] - tree.points[b][0], tree.points[a][1] - tree.points[b][1]);
    const drawnAt = new Float64Array(tree.points.length);

    for (const run of runs) {
      const speed = SETTINGS.speed / (1 + k * run.level);
      const start = drawnAt[run.seq[0]];
      let t = start;
      for (let i = 1; i < run.seq.length; i++) {
        t += len(run.seq[i - 1], run.seq[i]) / speed;
        drawnAt[run.seq[i]] = t;
      }
      run.start = start;
      run.dur = Math.max(0.001, t - start);
    }

    return {
      total: Math.max(...runs.map((r) => r.start + r.dur)),
      spineEnd: Math.max(...runs.filter((r) => r.level === 0).map((r) => r.start + r.dur)),
    };
  }

  // Once the skeleton is drawn there is nothing left to watch but the tail of
  // fine branches, so time itself speeds up, easing in rather than jumping.
  function warp(t, spineEnd, rate) {
    if (t <= spineEnd || rate <= 1) return t;
    const d = t - spineEnd;
    const gain = rate - 1;
    if (d <= SETTINGS.ramp) return spineEnd + d + (gain * d * d) / (2 * SETTINGS.ramp);
    return spineEnd + SETTINGS.ramp + (gain * SETTINGS.ramp) / 2 + rate * (d - SETTINGS.ramp);
  }

  function render(runs, t) {
    for (const run of runs) {
      const p = Math.min(1, Math.max(0, (t - run.start) / run.dur));
      run.el.style.strokeDashoffset = run.len * (1 - p);
    }
  }

  function build(frame) {
    const svg = frame.querySelector(".tree-frame-canvas");
    const box = frame.getBoundingClientRect();
    if (box.width < 10 || box.height < 10) return;

    // The drawing space is always 1000 units wide, with height following the
    // frame's real proportions, so the band keeps a constant thickness
    // relative to the width no matter how tall the content is.
    const w = 1000;
    const h = Math.round((box.height / box.width) * 1000);
    const band = SETTINGS.band;

    const shape = {
      w,
      h,
      inside: (x, y) => x < band || x > w - band || y < band || y > h - band,
      cut: (a, b) => (a[0] - w / 2) * (b[0] - w / 2) < 0 && a[1] > h - band && b[1] > h - band,
    };

    svg.setAttribute("viewBox", `0 0 ${w} ${h}`);

    const rand = mulberry32((Math.random() * 1e9) | 0);
    const tree = grow(poissonDisc(shape, SETTINGS.spacing, rand), shape);
    const runs = skeletonRuns(tree);
    const { total, spineEnd } = schedule(tree, runs);

    svg.replaceChildren();
    const layer = document.createElementNS(SVGNS, "g");
    svg.appendChild(layer);

    for (const run of runs) {
      const path = document.createElementNS(SVGNS, "path");
      path.setAttribute("class", "tree-branch");
      path.setAttribute(
        "d",
        `M${run.seq.map((i) => `${tree.points[i][0].toFixed(1)},${tree.points[i][1].toFixed(1)}`).join("L")}`,
      );
      layer.appendChild(path);
      run.el = path;
      run.len = path.getTotalLength();
      path.style.strokeDasharray = run.len;
      path.style.strokeDashoffset = run.len;
    }

    // Someone who has asked their system to reduce motion gets the finished
    // tree rather than no tree at all.
    if (matchMedia("(prefers-reduced-motion: reduce)").matches) {
      render(runs, total);
      return;
    }

    // The clock starts when the page is actually being looked at. Browsers
    // don't run animation frames in a hidden tab, so a page opened in a
    // background tab would otherwise stamp its start time on load and, by the
    // time anyone switched to it, jump straight to a fully grown tree.
    const play = () => {
      const started = performance.now();
      const step = (now) => {
        const t = warp((now - started) / 1000, spineEnd, SETTINGS.catchup);
        render(runs, t);
        if (t < total + 0.3) requestAnimationFrame(step);
      };
      requestAnimationFrame(step);
    };

    if (document.hidden) {
      document.addEventListener("visibilitychange", function once() {
        if (document.hidden) return;
        document.removeEventListener("visibilitychange", once);
        play();
      });
    } else {
      play();
    }
  }

  function init() {
    for (const frame of document.querySelectorAll(".tree-frame")) {
      const svg = frame.querySelector(".tree-frame-canvas");
      build(frame);

      // Resizing hides the tree and regrows it once the window has been
      // still for a moment.
      //
      // Hiding immediately is the important half: until the rebuild lands the
      // svg still holds the previous viewBox, and the browser fits that old
      // aspect ratio inside the new box, which makes the frame visibly shrink
      // inward, away from the edges it is meant to trace. Better to show
      // nothing for those frames than to show it wrong.
      //
      // Waiting for stillness is the other half: a window drag fires this
      // continuously, and regenerating a few hundred points on every frame of
      // that would be both slow and frantic. One regrowth, after the dragging
      // stops.
      let last = frame.getBoundingClientRect();
      let timer = null;
      new ResizeObserver(() => {
        const now = frame.getBoundingClientRect();
        const changed =
          Math.abs(now.width - last.width) >= SETTINGS.minChange ||
          Math.abs(now.height - last.height) >= SETTINGS.minChange;
        if (!changed) return;
        last = now;
        svg.classList.add("is-stale");
        clearTimeout(timer);
        timer = setTimeout(() => {
          build(frame);
          svg.classList.remove("is-stale");
        }, SETTINGS.resettle);
      }).observe(frame);
    }
  }

  if (document.readyState === "loading") document.addEventListener("DOMContentLoaded", init);
  else init();
})();
