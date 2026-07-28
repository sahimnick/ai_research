"""
visualizer.py
=============

A **live graph view** of the brain: watch the wave of excited neurons travel
from the sensory input, through the association regions, to the motor output.

The user asked to *see the path of triggered neurons, from entry to exit, live*
when a stimulus (sound / image / text / ...) is applied. This module builds a
graph of the network (neurons = nodes, synapses = edges), lays it out left
(input) to right (output), and animates it against a recorded run:

    * a neuron's colour/size shows how strongly and recently it fired,
    * an edge lights up while current is flowing along it,
    * the title shows the clock and which concepts the world model sees.

Two ways to watch:
    viz.animate(frames, live=True)            # interactive window (your machine)
    viz.animate(frames, save_path="run.gif")  # write a shareable GIF (headless)

For big networks a representative, well-connected subset of neurons is drawn so
the picture stays readable; ``mode="region"`` collapses each region to a single
node for a bird's-eye view.
"""

from __future__ import annotations

from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np

from ..core.brain import Brain, SpikeFrame


# Matplotlib is imported lazily inside methods so importing the package never
# forces a GUI backend. `Agg` is selected automatically when saving to a file.


class NetworkVisualizer:
    """Builds and animates a graph view of a :class:`Brain`.

    Parameters
    ----------
    brain:
        The brain to visualise (should be finalized).
    max_neurons_per_region:
        Cap on how many neurons of each region are drawn (keeps it legible).
    max_edges:
        Cap on how many synapses are drawn.
    seed:
        RNG seed for the sampling / node jitter, so pictures are reproducible.
    """

    def __init__(
        self,
        brain: Brain,
        max_neurons_per_region: int = 40,
        max_edges: int = 700,
        seed: int = 0,
    ):
        self.brain = brain
        self.max_per_region = int(max_neurons_per_region)
        self.max_edges = int(max_edges)
        self.rng = np.random.default_rng(seed)

        self.node_gids: List[int] = []
        self.gid_to_node: Dict[int, int] = {}
        self.node_pos: np.ndarray = np.zeros((0, 2))
        self.node_color: List[str] = []
        self.node_region: List[str] = []
        self.edges: List[Tuple[int, int]] = []       # (node_i, node_j)
        self._built = False

    # -- graph construction ----------------------------------------------
    def _region_of_gid(self, gid: int) -> str:
        for name, r in self.brain.regions.items():
            if r.gid_start <= gid < r.gid_start + r.size:
                return name
        return "?"

    def build(self) -> "NetworkVisualizer":
        """Select a connected, legible subset of neurons and their synapses."""
        chosen: Dict[str, set] = {n: set() for n in self.brain.regions}

        # 1. Walk edges in random order, keeping ones whose endpoints still fit,
        #    so the drawn neurons are actually connected across regions.
        kept_edges: List[Tuple[int, int]] = []
        proj = list(self.brain.projections)
        self.rng.shuffle(proj)
        for bundle in proj:
            src, dst = bundle.src_name, bundle.dst_name
            order = self.rng.permutation(len(bundle))
            for k in order:
                if len(kept_edges) >= self.max_edges:
                    break
                pre_l = int(bundle.pre[k])
                post_l = int(bundle.post[k])
                sp, dp = chosen[src], chosen[dst]
                pre_ok = pre_l in sp or len(sp) < self.max_per_region
                post_ok = post_l in dp or len(dp) < self.max_per_region
                if pre_ok and post_ok:
                    sp.add(pre_l)
                    dp.add(post_l)
                    g_pre = self.brain.regions[src].gid_start + pre_l
                    g_post = self.brain.regions[dst].gid_start + post_l
                    kept_edges.append((g_pre, g_post))
            if len(kept_edges) >= self.max_edges:
                break

        # 2. Make sure every region shows at least a few neurons.
        for name, r in self.brain.regions.items():
            want = min(self.max_per_region, r.size)
            while len(chosen[name]) < min(6, want):
                chosen[name].add(int(self.rng.integers(r.size)))

        # 3. Assign node ids and positions (region centre + jitter cloud).
        self.node_gids = []
        self.node_color = []
        self.node_region = []
        positions: List[Tuple[float, float]] = []
        for name, r in self.brain.regions.items():
            cx, cy = r.position if r.position else (0.5, 0.5)
            local = sorted(chosen[name])
            for li in local:
                gid = r.gid_start + li
                self.gid_to_node[gid] = len(self.node_gids)
                self.node_gids.append(gid)
                angle = self.rng.uniform(0, 2 * np.pi)
                rad = 0.10 * np.sqrt(self.rng.uniform(0, 1))
                positions.append((cx + rad * np.cos(angle),
                                  cy + rad * np.sin(angle)))
                inhib = r.population.inhibitory[li]
                self.node_color.append("#E8734C" if inhib else "#4C9BE8")
                self.node_region.append(name)
        self.node_pos = np.array(positions) if positions else np.zeros((0, 2))

        # 4. Keep only edges whose both endpoints are displayed.
        self.edges = [
            (self.gid_to_node[a], self.gid_to_node[b])
            for a, b in kept_edges
            if a in self.gid_to_node and b in self.gid_to_node
        ]
        self._built = True
        return self

    # -- animation --------------------------------------------------------
    def animate(
        self,
        frames: Sequence[SpikeFrame],
        save_path: Optional[str] = None,
        live: bool = False,
        interval: int = 80,
        title: str = "NeuroBrain — live activation path",
        fps: int = 12,
    ):
        """Animate a recorded run.

        Parameters
        ----------
        frames:
            The list returned by ``brain.run(...)`` / ``brain.present(...)``.
        save_path:
            If given, write an animated GIF/MP4 here (works without a display).
        live:
            If True, open an interactive window (use on your own machine).
        interval:
            Milliseconds between frames in the live window.
        """
        import matplotlib
        if save_path and not live:
            matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        from matplotlib.collections import LineCollection
        from matplotlib.animation import FuncAnimation, PillowWriter

        if not self._built:
            self.build()
        if not frames:
            raise ValueError("No frames to animate. Run the brain first.")

        fig, ax = plt.subplots(figsize=(11, 6.5))
        fig.patch.set_facecolor("#0e1117")
        ax.set_facecolor("#0e1117")
        ax.set_xlim(-0.2, 1.2)
        ax.set_ylim(-0.2, 1.2)
        ax.axis("off")

        # Region labels down the input->output axis.
        for name, r in self.brain.regions.items():
            cx, cy = r.position if r.position else (0.5, 0.5)
            ax.text(cx, cy + 0.16, name, color="#9aa4b2", ha="center",
                    fontsize=9, weight="bold")
            ax.text(cx, cy - 0.17, f"[{r.role}]", color="#5b6570",
                    ha="center", fontsize=7)

        # Static edge layer (dim), plus a dynamic layer we recolour per frame.
        seg = np.array([[self.node_pos[i], self.node_pos[j]]
                        for i, j in self.edges]) if self.edges else \
            np.zeros((0, 2, 2))
        base_edges = LineCollection(seg, colors="#20262f", linewidths=0.5,
                                    zorder=1)
        ax.add_collection(base_edges)
        live_edges = LineCollection(seg, colors=[(0, 0, 0, 0)] * len(self.edges),
                                    linewidths=1.6, zorder=2)
        ax.add_collection(live_edges)

        base_rgb = np.array([_hex_to_rgb(c) for c in self.node_color]) \
            if self.node_color else np.zeros((0, 3))
        scat = ax.scatter(self.node_pos[:, 0], self.node_pos[:, 1],
                          s=30, c=self.node_color, edgecolors="none", zorder=3)
        title_txt = ax.set_title(title, color="#e6e6e6", fontsize=12)

        edge_src = np.array([i for i, _ in self.edges], dtype=int) \
            if self.edges else np.zeros(0, dtype=int)

        def draw(fi: int):
            frame = frames[fi]
            act = frame.activation[self.node_gids] if len(self.node_gids) else \
                np.zeros(0)

            # Node colour: fade from base colour to hot white with activation.
            if len(act):
                a = np.clip(act, 0, 1)[:, None]
                hot = np.array([1.0, 0.95, 0.4])  # spiking glow
                rgb = base_rgb * 0.35 + (base_rgb * (1 - a) + hot * a) * 0.65
                scat.set_color(np.clip(rgb, 0, 1))
                scat.set_sizes(30 + 170 * a.ravel())

            # Edge glow: how active is each edge's source neuron right now.
            if len(self.edges):
                src_act = np.clip(act[edge_src], 0, 1) if len(act) else \
                    np.zeros(len(self.edges))
                ecol = np.zeros((len(self.edges), 4))
                ecol[:, 0] = 1.0            # R
                ecol[:, 1] = 0.85           # G
                ecol[:, 2] = 0.3            # B
                ecol[:, 3] = 0.15 + 0.85 * src_act  # alpha follows the signal
                live_edges.set_color(ecol)
                live_edges.set_linewidths(0.6 + 2.4 * src_act)

            concepts = ", ".join(frame.active_concepts[:4]) or "—"
            title_txt.set_text(
                f"{title}\nt = {frame.t} ms    active concepts: {concepts}"
            )
            return scat, live_edges, title_txt

        anim = FuncAnimation(fig, draw, frames=len(frames),
                             interval=interval, blit=False)

        if save_path:
            if save_path.lower().endswith(".gif"):
                anim.save(save_path, writer=PillowWriter(fps=fps))
            else:
                anim.save(save_path, fps=fps)
            plt.close(fig)
            return save_path
        if live:
            plt.show()
        return anim

    # -- a quick, dependency-free text trace -----------------------------
    def print_trace(self, frames: Sequence[SpikeFrame], every: int = 5) -> None:
        """ASCII summary of a run: firing rate per region over time.

        Handy when you have no display and just want to confirm the signal
        propagated input -> output.
        """
        names = list(self.brain.regions)
        print("t    | " + " | ".join(f"{n[:10]:>10}" for n in names)
              + " | concepts")
        print("-" * (8 + 13 * len(names) + 12))
        for fi, frame in enumerate(frames):
            if fi % every:
                continue
            cells = []
            for name in names:
                r = self.brain.regions[name]
                a = frame.activation[r.gid_start:r.gid_start + r.size].mean()
                bar = "█" * int(a * 10)
                cells.append(f"{bar:>10}")
            concepts = ",".join(frame.active_concepts[:3])
            print(f"{frame.t:4d} | " + " | ".join(cells) + f" | {concepts}")


def _hex_to_rgb(h: str) -> Tuple[float, float, float]:
    h = h.lstrip("#")
    return tuple(int(h[i:i + 2], 16) / 255.0 for i in (0, 2, 4))
