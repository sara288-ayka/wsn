from flask import Flask, render_template, request
import numpy as np
import nashpy as nash

app = Flask(__name__)


@app.route("/", methods=["GET", "POST"])
def index():
    output = None

    if request.method == "POST":
        try:
            num_strategies = int(request.form.get("num_strategies", 2))
            num_strategies = max(2, min(4, num_strategies))

            sensor1_name = request.form.get("sensor1_name", "Capteur S1").strip() or "Capteur S1"
            sensor2_name = request.form.get("sensor2_name", "Capteur S2").strip() or "Capteur S2"

            route_names = []
            for i in range(num_strategies):
                rn = request.form.get(f"route{i+1}_name", f"Route {i+1}").strip()
                route_names.append(rn or f"Route {i+1}")

            n = num_strategies
            mat_a, mat_b = [], []
            for r in range(n):
                row_a, row_b = [], []
                for c in range(n):
                    va = request.form.get(f"a{r}{c}", "0") or "0"
                    vb = request.form.get(f"b{r}{c}", "0") or "0"
                    row_a.append(float(va))
                    row_b.append(float(vb))
                mat_a.append(row_a)
                mat_b.append(row_b)

            arr_a = np.array(mat_a)
            arr_b = np.array(mat_b)

            game = nash.Game(arr_a, arr_b)
            try:
                eq_list = list(game.support_enumeration())
            except Exception:
                eq_list = []

            all_outcomes, route_labels = [], []
            for r in range(n):
                for c in range(n):
                    all_outcomes.append((mat_a[r][c], mat_b[r][c]))
                    route_labels.append((route_names[r], route_names[c]))

            pareto_keep = []
            for i, p in enumerate(all_outcomes):
                dominated = any(
                    q[0] >= p[0] and q[1] >= p[1] and (q[0] > p[0] or q[1] > p[1])
                    for q in all_outcomes
                )
                if not dominated:
                    pareto_keep.append({"s1": p[0], "s2": p[1],
                                        "r1": route_labels[i][0], "r2": route_labels[i][1]})

            def find_dominant(mat, row_player):
                sz = len(mat)
                for s in range(sz):
                    if all(
                        (mat[s][c] > mat[o][c] if row_player else mat[r][s] > mat[r][o])
                        for o in range(sz) if o != s
                        for c in ([c for c in range(sz)] if row_player else [])
                        for r in ([] if row_player else [r for r in range(sz)])
                    ):
                        return s
                # simpler explicit check
                for s in range(sz):
                    dom = True
                    for o in range(sz):
                        if o == s: continue
                        if row_player:
                            if not all(mat[s][c] > mat[o][c] for c in range(sz)):
                                dom = False; break
                        else:
                            if not all(mat[r][s] > mat[r][o] for r in range(sz)):
                                dom = False; break
                    if dom:
                        return s
                return None

            dom_s1 = find_dominant(mat_a, True)
            dom_s2 = find_dominant(mat_b, False)

            steps, pace = 100, 0.12
            px = np.ones(n) / n
            py = np.ones(n) / n
            hist_x, hist_y, energy_hist = [], [], []

            for _ in range(steps):
                pay_x = arr_a @ py
                pay_y = arr_b.T @ px
                avg_px = float(px @ pay_x)
                avg_py = float(py @ pay_y)
                energy_hist.append(round(avg_px + avg_py, 4))
                px = np.clip(px + pace * px * (pay_x - avg_px), 0, None)
                py = np.clip(py + pace * py * (pay_y - avg_py), 0, None)
                px = px / px.sum() if px.sum() > 0 else np.ones(n)/n
                py = py / py.sum() if py.sum() > 0 else np.ones(n)/n
                hist_x.append([round(float(v), 4) for v in px])
                hist_y.append([round(float(v), 4) for v in py])

            output = {
                "nash": [(list(map(float, e[0])), list(map(float, e[1]))) for e in eq_list],
                "pareto": pareto_keep,
                "final_x": hist_x[-1],
                "final_y": hist_y[-1],
                "hist_x": hist_x,
                "hist_y": hist_y,
                "energy_hist": energy_hist,
                "sensor1_name": sensor1_name,
                "sensor2_name": sensor2_name,
                "route_names": route_names,
                "num_strategies": n,
                "dominant_s1": route_names[dom_s1] if dom_s1 is not None else None,
                "dominant_s2": route_names[dom_s2] if dom_s2 is not None else None,
                "mat_a": mat_a,
                "mat_b": mat_b,
            }
        except Exception as e:
            output = {"error": str(e)}

    return render_template("index.html", output=output)


if __name__ == "__main__":
    app.run(debug=True)
