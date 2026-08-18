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
    bandPx: 41, // band thickness on screen, in css pixels
    spacing: 9, // minimum distance between scattered points
    turn: 2, // how much a change of direction costs when growing
    slowdown: 0.25, // speed of each level of branch relative to the one above
    catchup: 3.5, // how much time accelerates once the skeleton is drawn
    speed: 130, // svg units per second, for the ring
    ribbonSpeed: 85, // slower on the narrow layout, where each run is shorter
    ramp: 0.8, // seconds over which catch-up eases in
    resettle: 800, // ms of stillness after a resize before the tree regrows
    minChange: 24, // px of size change worth regenerating for
    maxPoints: 1400, // ceiling on scatter size, whatever shape the frame is
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

    // Start from the region's own seed. Anywhere outside its mask and every
    // candidate around it is rejected, leaving the region empty.
    const first = shape.seed;
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
    const seed = shape.seed;

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
  function schedule(tree, runs, speed) {
    const k = 1 / SETTINGS.slowdown - 1;
    const len = (a, b) => Math.hypot(tree.points[a][0] - tree.points[b][0], tree.points[a][1] - tree.points[b][1]);
    const drawnAt = new Float64Array(tree.points.length);

    for (const run of runs) {
      const runSpeed = speed / (1 + k * run.level);
      const start = drawnAt[run.seq[0]];
      let t = start;
      for (let i = 1; i < run.seq.length; i++) {
        t += len(run.seq[i - 1], run.seq[i]) / runSpeed;
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

  // ---------------------------------------------------------------------
  // A tree is grown from a spec: a region to fill and how to fill it.
  //
  //   inside(x, y)  is this point part of the region
  //   seed          where growth starts, and where the scatter begins
  //   cut(a, b)     edges to forbid, used to break a loop at a chosen place
  //   speed         units per second for the skeleton
  //   spacing       minimum distance between scattered points
  //   area          how much of the canvas the region covers, for point budgets
  //
  // Everything a tree needs is in the spec, so growing several is just calling
  // this more than once. They have to be separate calls: Prim's spans whatever
  // points it is given, so one pass over two bands would join them with a
  // single long edge straight across the content between them.
  // ---------------------------------------------------------------------
  function growTree(spec, rand) {
    const tree = grow(poissonDisc(spec, spec.spacing, rand), spec);
    const runs = skeletonRuns(tree);
    const { total, spineEnd } = schedule(tree, runs, spec.speed);
    return { points: tree.points, runs, total, spineEnd };
  }

  // A band of the given thickness around the edge of the canvas. It is a loop,
  // so it carries a cut at the bottom centre, opposite the seed: a tree cannot
  // hold a loop, and left to itself Prim's breaks the ring at an arbitrary
  // edge, sending growth most of the way round one side. Cutting it opposite
  // the seed splits it into halves that meet at the bottom.
  const ringSpec = (w, h, band) => ({
    w,
    h,
    inside: (x, y) => x < band || x > w - band || y < band || y > h - band,
    seed: [w / 2, band / 2],
    cut: (a, b) => (a[0] - w / 2) * (b[0] - w / 2) < 0 && a[1] > h - band && b[1] > h - band,
    speed: SETTINGS.speed,
    area: w * h - Math.max(0, w - 2 * band) * Math.max(0, h - 2 * band),
  });

  // A horizontal band across the top or the bottom. Two ends, no loop, so
  // nothing to cut.
  const ribbonSpec = (w, h, band, edge) => ({
    w,
    h,
    inside: (x, y) => (edge === "top" ? y < band : y > h - band),
    seed: [w / 2, edge === "top" ? band / 2 : h - band / 2],
    cut: () => false,
    speed: SETTINGS.ribbonSpeed,
    area: w * band,
  });

  // Which specs to grow. The stylesheet decides, via `--tree-shape` on the
  // frame, so the breakpoint lives with the styling rather than being
  // duplicated as a number here.
  function specsFor(frame, w, h, band) {
    const layout = getComputedStyle(frame).getPropertyValue("--tree-shape").trim();

    if (layout === "ribbons") return [ribbonSpec(w, h, band, "top"), ribbonSpec(w, h, band, "bottom")];
    return [ringSpec(w, h, band)];
  }

  function build(frame) {
    const svg = frame.querySelector(".tree-frame-canvas");
    const box = frame.getBoundingClientRect();
    if (box.width < 10 || box.height < 10) return;

    // The stylesheet can hide the canvas; building a tree nobody can see is
    // pure cost.
    if (getComputedStyle(svg).display === "none") return;

    // The drawing space is always 1000 units wide, with height following the
    // frame's real proportions, so a band keeps a constant thickness relative
    // to the width no matter how tall the content is.
    const w = 1000;
    const h = Math.round((box.height / box.width) * 1000);

    // Band thickness is specified on screen rather than in drawing units. The
    // drawing space is always 1000 units wide, so a fixed number of units is a
    // fixed *fraction* of the width -- which on a phone came out half as thick
    // as on a desktop and looked spindly. Converting from pixels keeps it the
    // same weight at every size.
    const band = (SETTINGS.bandPx / box.width) * 1000;

    const specs = specsFor(frame, w, h, band);

    // Spacing is defined against a 1000-unit-wide space, so a tall narrow
    // frame maps to a viewBox thousands of units deep and the area to fill
    // grows with it. Left unbounded that took the scatter from ~440 points to
    // 3772 and the build from 65ms to 1.9 seconds, freezing the page. The
    // budget is shared across every spec, and each is thinned to fit.
    // ~1.49 * spacing^2 of area per point, measured from the blue-noise scatter.
    const area = specs.reduce((sum, spec) => sum + spec.area, 0);
    const estimate = area / (1.49 * SETTINGS.spacing * SETTINGS.spacing);
    const spacing = estimate > SETTINGS.maxPoints ? Math.sqrt(area / (1.49 * SETTINGS.maxPoints)) : SETTINGS.spacing;
    for (const spec of specs) spec.spacing = spacing;

    svg.setAttribute("viewBox", `0 0 ${w} ${h}`);
    svg.replaceChildren();
    const layer = document.createElementNS(SVGNS, "g");
    svg.appendChild(layer);

    const rand = mulberry32((Math.random() * 1e9) | 0);
    const grown = specs.map((spec) => growTree(spec, rand));

    // Every tree is drawn into the same canvas and played on one clock. Their
    // timelines all start at zero, so they grow alongside each other.
    const drawn = grown.flatMap((tree) => draw(layer, tree));
    const total = Math.max(0.001, ...grown.map((tree) => tree.total));
    const spineEnd = Math.max(0, ...grown.map((tree) => tree.spineEnd));

    // Someone who has asked their system to reduce motion gets the finished
    // tree rather than no tree at all.
    if (matchMedia("(prefers-reduced-motion: reduce)").matches) {
      render(drawn, total);
      return;
    }

    const started = performance.now();
    const step = (now) => {
      const t = warp((now - started) / 1000, spineEnd, SETTINGS.catchup);
      render(drawn, t);
      if (t < total + 0.3) requestAnimationFrame(step);
    };
    requestAnimationFrame(step);
  }

  // One path per run, hidden by its own dash offset until the clock reaches it.
  function draw(layer, { points, runs }) {
    return runs.map((run) => {
      // Round once, then use the same numbers for the path data and for its
      // length. getTotalLength() would give the same answer but forces a
      // layout per path, and there are several hundred of them.
      const pts = run.seq.map((i) => [+points[i][0].toFixed(1), +points[i][1].toFixed(1)]);
      let len = 0;
      for (let i = 1; i < pts.length; i++) len += Math.hypot(pts[i][0] - pts[i - 1][0], pts[i][1] - pts[i - 1][1]);

      const path = document.createElementNS(SVGNS, "path");
      path.setAttribute("class", "tree-branch");
      path.setAttribute("d", `M${pts.map((p) => `${p[0]},${p[1]}`).join("L")}`);
      layer.appendChild(path);

      path.style.strokeDasharray = len;
      path.style.strokeDashoffset = len;
      return { el: path, len, start: run.start, dur: run.dur };
    });
  }

  // Wait until the frame can be measured correctly.
  //
  // A hidden tab first: browsers run no animation frames there, so a page
  // opened in a background tab would otherwise stamp its start time on load
  // and, by the time anyone looked, jump straight to a grown tree.
  //
  // Then `load` and the webfont swap. Vollkorn changes this frame's height by
  // ~30px against the fallback, so a tree built before the swap is drawn to
  // the wrong shape -- and the reflow would trip the resize handler, hiding
  // the tree and regrowing it a beat later.
  async function settled() {
    if (document.hidden) {
      await new Promise((resolve) => {
        document.addEventListener("visibilitychange", function once() {
          if (document.hidden) return;
          document.removeEventListener("visibilitychange", once);
          resolve();
        });
      });
    }

    if (document.readyState !== "complete") {
      await new Promise((resolve) => addEventListener("load", resolve, { once: true }));
    }

    // Optional and failure-tolerant: a missing or rejected font promise is not
    // worth refusing to draw over.
    await Promise.allSettled([document.fonts ? document.fonts.ready : null]);
  }

  async function init() {
    const frames = [...document.querySelectorAll(".tree-frame")];
    if (!frames.length) return;

    // Wait before measuring anything. The webfont swap alone changes this
    // frame's height by ~30px, so a tree built before it would be drawn to
    // the wrong shape -- and the reflow would trip the resize handler below,
    // hiding the tree and regrowing it a beat later.
    await settled();

    for (const frame of frames) {
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
