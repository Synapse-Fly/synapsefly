"use client";
// A live 3D fly brain: each of the 8 regions is a point cloud of neurons, arranged in a stylized
// Drosophila-brain layout; every point's brightness tracks its region's live firing rate
// (tick.rates.regions), and the pathway lines between regions pulse with the activity flowing along
// them. Orbit with the mouse; auto-rotates when idle. Three.js is dynamically imported (client only).
import { useEffect, useRef, useState } from "react";
import type { FlySocket } from "@/lib/ws";

// REGIONS order (SPEC 0.1): optic_lobe, antennal_lobe, mushroom_body, central_complex, sez,
// central_other, descending_motor, vnc. Each entry: base color + one or two anatomical centers
// (paired regions get L/R) + a relative neuron share used to size its point cloud.
interface RegionDef { name: string; color: [number, number, number]; centers: [number, number, number][]; share: number }
const REGIONS: RegionDef[] = [
  { name: "optic lobe",       color: [0.13, 0.83, 0.93], centers: [[-3.1, 0.4, 0.6], [3.1, 0.4, 0.6]], share: 0.42 },
  { name: "antennal lobe",    color: [0.85, 0.28, 0.94], centers: [[-0.9, -0.7, 1.5], [0.9, -0.7, 1.5]], share: 0.05 },
  { name: "mushroom body",    color: [0.92, 0.70, 0.05], centers: [[-1.4, 1.3, 0.1], [1.4, 1.3, 0.1]], share: 0.12 },
  { name: "central complex",  color: [0.13, 0.77, 0.37], centers: [[0.0, 0.6, -0.3]], share: 0.05 },
  { name: "SEZ",              color: [0.98, 0.45, 0.09], centers: [[0.0, -1.4, 0.4]], share: 0.10 },
  { name: "central (other)",  color: [0.58, 0.64, 0.72], centers: [[0.0, 0.2, 0.1]], share: 0.13 },
  { name: "descending/motor", color: [0.94, 0.27, 0.27], centers: [[0.0, -1.9, -0.9]], share: 0.05 },
  { name: "VNC",              color: [0.86, 0.15, 0.15], centers: [[0.0, -3.4, -1.1], [0.0, -4.6, -1.3]], share: 0.08 },
];
// Representative pathways (region index pairs); each line's glow = mean of the two regions' rates.
const PATHWAYS: [number, number][] = [
  [0, 3], [0, 6], [1, 2], [2, 5], [3, 6], [4, 6], [5, 3], [6, 7],
];
const TOTAL_POINTS = 14000;
const RATE_FULL_HZ = 18; // firing rate that maps to full brightness

function centroid(d: RegionDef): [number, number, number] {
  const c = d.centers;
  return [c.reduce((s, p) => s + p[0], 0) / c.length, c.reduce((s, p) => s + p[1], 0) / c.length, c.reduce((s, p) => s + p[2], 0) / c.length];
}

export default function BrainView3D({ sock }: { sock: FlySocket }) {
  const mountRef = useRef<HTMLDivElement | null>(null);
  const [err, setErr] = useState<string | null>(null);

  useEffect(() => {
    const mount = mountRef.current;
    if (!mount) return;
    let disposed = false;
    let cleanup = () => {};

    (async () => {
      let THREE: typeof import("three");
      let OrbitControls: typeof import("three/examples/jsm/controls/OrbitControls.js")["OrbitControls"];
      try {
        THREE = await import("three");
        ({ OrbitControls } = await import("three/examples/jsm/controls/OrbitControls.js"));
      } catch (e) {
        setErr("3D unavailable");
        console.error("[brain3d] three import failed", e);
        return;
      }
      if (disposed) return;

      const w = mount.clientWidth || 400, h = mount.clientHeight || 400;
      const scene = new THREE.Scene();
      scene.background = new THREE.Color(0x05070a);
      const camera = new THREE.PerspectiveCamera(50, w / h, 0.1, 100);
      camera.position.set(0, -0.5, 11);
      const renderer = new THREE.WebGLRenderer({ antialias: true, alpha: false });
      renderer.setPixelRatio(Math.min(2, window.devicePixelRatio || 1));
      renderer.setSize(w, h);
      mount.appendChild(renderer.domElement);

      const controls = new OrbitControls(camera, renderer.domElement);
      controls.enableDamping = true;
      controls.dampingFactor = 0.08;
      controls.autoRotate = true;
      controls.autoRotateSpeed = 0.8;
      controls.enablePan = false;
      controls.minDistance = 5;
      controls.maxDistance = 24;
      controls.target.set(0, -0.8, 0);

      // ---- build the point cloud (positions + per-region index + base colors) ----
      const counts = sock.hello?.connectome.region_counts ?? null;
      const shares = REGIONS.map((r, i) => {
        const key = (sock.hello?.regions?.[i]) as string | undefined;
        const c = counts && key ? (counts as Record<string, number>)[key] : undefined;
        return (typeof c === "number" && c > 0) ? c : r.share * 100000;
      });
      const shareSum = shares.reduce((s, v) => s + v, 0);
      const perRegion = shares.map((s) => Math.max(120, Math.round((s / shareSum) * TOTAL_POINTS)));
      const N = perRegion.reduce((s, v) => s + v, 0);

      const positions = new Float32Array(N * 3);
      const colors = new Float32Array(N * 3);
      const baseColors = new Float32Array(N * 3);
      const regionOf = new Uint8Array(N);
      const gauss = () => (Math.random() + Math.random() + Math.random() - 1.5) * 0.9;
      let p = 0;
      REGIONS.forEach((r, ri) => {
        const n = perRegion[ri];
        const spread = ri === 0 ? 0.85 : ri === 7 ? [0.5, 0.9, 0.5] : 0.55; // optic wide, VNC elongated
        for (let k = 0; k < n; k++) {
          const c = r.centers[k % r.centers.length];
          const sx = Array.isArray(spread) ? spread[0] : spread;
          const sy = Array.isArray(spread) ? spread[1] : spread;
          const sz = Array.isArray(spread) ? spread[2] : spread;
          positions[p * 3] = c[0] + gauss() * sx;
          positions[p * 3 + 1] = c[1] + gauss() * sy;
          positions[p * 3 + 2] = c[2] + gauss() * sz;
          baseColors[p * 3] = r.color[0]; baseColors[p * 3 + 1] = r.color[1]; baseColors[p * 3 + 2] = r.color[2];
          regionOf[p] = ri;
          p++;
        }
      });
      colors.set(baseColors);
      const geom = new THREE.BufferGeometry();
      geom.setAttribute("position", new THREE.BufferAttribute(positions, 3));
      geom.setAttribute("color", new THREE.BufferAttribute(colors, 3));
      const mat = new THREE.PointsMaterial({ size: 0.055, vertexColors: true, transparent: true, opacity: 0.9, sizeAttenuation: true, depthWrite: false, blending: THREE.AdditiveBlending });
      const points = new THREE.Points(geom, mat);
      scene.add(points);

      // ---- pathway lines between region centroids ----
      const cents = REGIONS.map(centroid);
      const lineObjs = PATHWAYS.map(([a, b]) => {
        const g = new THREE.BufferGeometry().setFromPoints([
          new THREE.Vector3(...cents[a]), new THREE.Vector3(...cents[b]),
        ]);
        const m = new THREE.LineBasicMaterial({ color: 0x2ee66a, transparent: true, opacity: 0.15 });
        const line = new THREE.Line(g, m);
        scene.add(line);
        return { line, a, b, m };
      });

      // ---- resize ----
      const onResize = () => {
        const ww = mount.clientWidth || 400, hh = mount.clientHeight || 400;
        camera.aspect = ww / hh; camera.updateProjectionMatrix(); renderer.setSize(ww, hh);
      };
      const ro = new ResizeObserver(onResize);
      ro.observe(mount);

      // ---- animation: recolor points by live region rate ----
      const colAttr = geom.getAttribute("color") as import("three").BufferAttribute;
      let raf = 0;
      const intensity = new Float32Array(8).fill(0.25);
      const animate = () => {
        if (disposed) return;
        raf = requestAnimationFrame(animate);
        const tick = sock.store.latest()?.tick;
        const rates = tick?.rates.regions;
        // smooth toward the live rate (EMA) so pulses read as glow, not flicker
        for (let i = 0; i < 8; i++) {
          const target = rates && Number.isFinite(rates[i]) ? Math.min(1, rates[i] / RATE_FULL_HZ) : 0;
          intensity[i] += (Math.max(0.18, target) - intensity[i]) * 0.12;
        }
        const arr = colAttr.array as Float32Array;
        for (let i = 0; i < N; i++) {
          const ri = regionOf[i];
          const g = 0.35 + 0.9 * intensity[ri];
          arr[i * 3] = Math.min(1, baseColors[i * 3] * g);
          arr[i * 3 + 1] = Math.min(1, baseColors[i * 3 + 1] * g);
          arr[i * 3 + 2] = Math.min(1, baseColors[i * 3 + 2] * g);
        }
        colAttr.needsUpdate = true;
        for (const lo of lineObjs) lo.m.opacity = 0.08 + 0.6 * (0.5 * (intensity[lo.a] + intensity[lo.b]));
        controls.update();
        renderer.render(scene, camera);
      };
      animate();

      cleanup = () => {
        cancelAnimationFrame(raf);
        ro.disconnect();
        controls.dispose();
        geom.dispose(); mat.dispose();
        for (const lo of lineObjs) { lo.line.geometry.dispose(); (lo.line.material as import("three").Material).dispose(); }
        renderer.dispose();
        if (renderer.domElement.parentNode === mount) mount.removeChild(renderer.domElement);
      };
    })();

    return () => { disposed = true; cleanup(); };
  }, [sock]);

  return (
    <div className="relative h-full w-full bg-[#05070a]">
      <div ref={mountRef} className="h-full w-full" />
      {/* legend */}
      <div className="pointer-events-none absolute left-1 top-1 flex flex-col gap-[1px] text-[10px] text-white/85">
        {REGIONS.map((r) => (
          <div key={r.name} className="flex items-center gap-1">
            <span className="inline-block h-[8px] w-[8px]" style={{ background: `rgb(${r.color.map((c) => Math.round(c * 255)).join(",")})` }} />
            {r.name}
          </div>
        ))}
      </div>
      <div className="pointer-events-none absolute bottom-1 right-2 text-[9px] text-white/50">drag to rotate · live firing</div>
      {err ? <div className="absolute inset-0 flex items-center justify-center text-[11px] text-white/70">{err}</div> : null}
    </div>
  );
}
