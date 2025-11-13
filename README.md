<h1 align="center">ProfileSOM</h1>
<p align="center">
  <strong>Self-organizing maps (SOM) with k-means clustering for geochemical profile data</strong>
</p>

<p align="center">
  <img src="logo.png" width="260"><br>
  <em>Hexagonal SOM + k-means (auto-K via Davies–Bouldin) on a BMU-hit-weighted codebook, with density-aware diagnostics</em>
</p>

<p align="center">
  <a href="LICENSE"><img alt="License" src="https://img.shields.io/badge/License-MIT-green.svg"></a>
  <img alt="Python" src="https://img.shields.io/badge/Python-3.9%2B-blue.svg">
</p>

<hr>

<h2>📚 Table of Contents</h2>
<ul>
  <li><a href="#overview">Overview</a></li>
  <li><a href="#features">Features</a></li>
  <li><a href="#installation">Installation</a></li>
  <li><a href="#quickstart">Quickstart</a></li>
  <li><a href="#cli">Command-line Arguments</a></li>
  <li><a href="#io">Input Format & Numeric Coercion</a></li>
  <li><a href="#outputs">Outputs</a></li>
  <li><a href="#figures">How to Read the Figures</a></li>
  <li><a href="#reproducibility">Reproducibility</a></li>
  <li><a href="#tips">Tips & Best Practices</a></li>
  <li><a href="#troubleshooting">Troubleshooting</a></li>
  <li><a href="#reuse">Re-using Components Programmatically</a></li>
  <li><a href="#citation">Citation & License</a></li>
</ul>

<hr>

<h2 id="overview">📄 Overview</h2>
<p>
  <strong>Hexagonal SOM + k-means</strong> trains a hexagonal Self-Organizing Map (SOM) on z-scored features, clusters the SOM codebook with k-means, and selects <code>k</code> by minimizing the <strong>Davies–Bouldin (DB)</strong> index computed on a <strong>sample-weighted codebook</strong>. Each prototype is replicated by its BMU hit count so that K selection reflects data density while avoiding per-sample BMU noise.
</p>

<hr>

<h2 id="features">⚙️ Features</h2>
<ul>
  <li><strong>Hex SOM topology</strong> with clean, publication-ready maps (U-matrix, hits, clusters, component planes, PCA biplot).</li>
  <li><strong>Principled K selection</strong>: DB on a <em>weighted</em> codebook emphasizes occupied neurons.</li>
  <li><strong>Robust I/O</strong>: Excel with header on row 2; smart ID detection; detection-limit coercion (<code>&lt;0.12</code>, <code>&lt;LD</code>).</li>
  <li><strong>Reproducibility</strong>: logs, environment snapshot, serialized artifacts (SOM weights, scaler, k-means, config).</li>
  <li><strong>Turnkey outputs</strong>: CSV summaries and neatly organized figures for immediate use in papers.</li>
</ul>

<hr>

<h2 id="installation">🚀 Installation</h2>
<pre><code># optional: create a virtual environment
python -m venv .venv
source .venv/bin/activate   # (Windows) .venv\Scripts\activate

pip install -U pip
pip install numpy pandas matplotlib minisom scikit-learn
</code></pre>

<hr>

<h2 id="quickstart">⚡ Quickstart</h2>
<pre><code>python profilesom.py --xlsx data.xlsx --sheet Data \
    --som_m 6 --som_n 6 --som_iters 1000 --k_min 2 --k_max 10 \
    --id_col "Sample" --depth_col "h" --profile_col "Profile"
</code></pre>

<hr>

<h2 id="cli">🧰 Command-line Arguments</h2>

<table>
<thead>
<tr><th>Flag</th><th>Type / Default</th><th>Description</th></tr>
</thead>
<tbody>
<tr><td><code>--xlsx</code></td><td><strong>required</strong></td><td>Path to Excel file (header on row 2).</td></tr>
<tr><td><code>--sheet</code></td><td><code>None</code></td><td>Sheet name (defaults to the first sheet).</td></tr>
<tr><td><code>--out_dir</code></td><td><code>outputs</code></td><td>Root output directory.</td></tr>
<tr><td><code>--som_m</code>, <code>--som_n</code></td><td><code>10</code>, <code>10</code></td><td>SOM grid size (rows × cols), hex topology.</td></tr>
<tr><td><code>--som_iters</code></td><td><code>1500</code></td><td>SOM training iterations.</td></tr>
<tr><td><code>--sigma</code></td><td><code>1.2</code></td><td>SOM neighborhood sigma.</td></tr>
<tr><td><code>--lr</code></td><td><code>0.5</code></td><td>SOM learning rate.</td></tr>
<tr><td><code>--k_min</code>, <code>--k_max</code></td><td><code>2</code>, <code>9</code></td><td>Inclusive scan range for k-means clusters.</td></tr>
<tr><td><code>--k_inits</code></td><td><code>50</code></td><td>k-means <code>n_init</code> per K.</td></tr>
<tr><td><code>--seed</code></td><td><code>42</code></td><td>Random seed.</td></tr>
<tr><td><code>--k</code></td><td><code>-1</code></td><td>Optional extra fixed K to also render/compare.</td></tr>
<tr><td><code>--u_center</code></td><td><code>builtin</code></td><td>Dense U-matrix center mode: <code>builtin</code> | <code>mean</code> | <code>median</code>.</td></tr>
<tr><td><code>--depth_col</code></td><td><code>h</code></td><td>Depth/height column (optional, for profile plots).</td></tr>
<tr><td><code>--profile_col</code></td><td><code>""</code></td><td>Profile/borehole ID column (optional).</td></tr>
<tr><td><code>--depth_bins</code></td><td><code>12</code></td><td>Reserved; not required for current plots.</td></tr>
<tr><td><code>--id_col</code></td><td><code>""</code></td><td>Force a specific sample ID column (overrides autodetect).</td></tr>
</tbody>
</table>

<hr>

<h2 id="io">🧾 Input Format & Numeric Coercion</h2>
<ul>
  <li>Excel file where the header is on <strong>row 2</strong> (<code>header=1</code> in pandas).</li>
  <li>All columns are considered features except:
    <ul>
      <li><code>h</code> (case-insensitive) — depth/height for optional profile plots,</li>
      <li>an optional profile/borehole column,</li>
      <li>the sample ID column (autodetected or passed via <code>--id_col</code>).</li>
    </ul>
  </li>
  <li><strong>Detection limits:</strong>
    <ul>
      <li>Strings like <code>&lt;0.12</code> or <code>&lt; 0,12</code> → half the limit (e.g., 0.06).</li>
      <li><code>&lt;LD</code> and <code>NaN</code> → half of the finite minimum observed for that column (if any).</li>
      <li>Remaining non-finite values are imputed by column median; all features are z-scored.</li>
    </ul>
  </li>
</ul>

<hr>

<h2 id="outputs">🗂 Outputs</h2>
<pre><code>outputs/
├─ log/
│  ├─ run.log
│  └─ environment.txt
├─ data/
│  ├─ som_sample_assignments.csv
│  ├─ som_codebook_vectors.csv
│  ├─ cluster_feature_summary.csv
│  ├─ cluster_sizes_db.csv
│  ├─ som_weights.npy
│  ├─ scaler.pkl
│  ├─ kmeans_best.pkl
│  └─ run_config.json
├─ diagnostics/
│  ├─ k_selection_metrics_weighted.csv
│  ├─ k_db_weighted.png / .pdf
│  └─ codebook_pca_biplot_k*.pdf
└─ plots/
   ├─ umatrix/      # dense U-matrix
   ├─ hits/         # hits maps (with/without IDs)
   ├─ clusters/     # per-k maps + best-DB map
   ├─ components/   # panel + per-feature planes
   ├─ profile/      # vertical cluster strips (if depth)
   └─ boxes/        # per-feature distributions
</code></pre>

<hr>

<h2 id="figures">🖼️ How to Read the Figures</h2>
<ul>
  <li><strong>U-matrix (dense hex):</strong> relief map of prototype distances; ridges ≈ cluster boundaries.</li>
  <li><strong>Hits map:</strong> occupancy per neuron; numbers inside cells (density sanity check).</li>
  <li><strong>Cluster maps (per k / best DB):</strong> discrete IDs with pixel-perfect boundaries.</li>
  <li><strong>Component planes:</strong> per-feature min–max maps; final panel shows clusters.</li>
  <li><strong>PCA biplot (codebook):</strong> prototypes in PC space colored by cluster; loadings arrows labeled at tips.</li>
  <li><strong>Box plots:</strong> per-cluster feature distributions (single or grouped panels).</li>
  <li><strong>Profile strips:</strong> depth-wise cluster blocks with glued IDs at the same level (optional).</li>
</ul>

<hr>

<h2 id="reproducibility">🧪 Reproducibility</h2>
<ul>
  <li>BLAS thread control to avoid oversubscription: <code>OMP_NUM_THREADS=1</code>, <code>MKL_NUM_THREADS=1</code> (set in code).</li>
  <li>Automatic environment snapshot: <code>log/environment.txt</code>.</li>
  <li>Serialized artifacts: SOM weights, scaler, and best k-means model.</li>
</ul>

<hr>

<h2 id="tips">💡 Tips & Best Practices</h2>
<ul>
  <li><strong>Grid size:</strong> start with <code>--som_m 6 --som_n 6</code> for small datasets; scale up as samples grow.</li>
  <li><strong>K range:</strong> keep <code>k_max ≤ 12</code> to avoid oversegmentation; DB penalizes too many clusters.</li>
  <li><strong>Features:</strong> remove near-constants and obvious categorical strings before export (script also drops non-informative columns).</li>
  <li><strong>Depth units:</strong> any numeric scale works; vertical plots auto-scale.</li>
</ul>

<hr>

<h2 id="troubleshooting">🛠 Troubleshooting</h2>
<ul>
  <li><strong>“No feature columns found”</strong> — Your sheet may contain only IDs or non-numeric content; ensure quantitative columns exist beyond <code>SampleID</code>, <code>h</code>, and <code>Profile</code>.</li>
  <li><strong>“Not enough usable feature columns”</strong> — After dropping constants/NaNs, fewer than 2 numeric columns remained; provide more variables.</li>
  <li><strong>Blank or flat plots</strong> — Inspect detection-limit strings (<code>&lt;LD</code>) and numeric coercion; columns with no numeric values are dropped.</li>
  <li><strong>Sparse hits</strong> — Reduce the grid or increase <code>--som_iters</code>.</li>
</ul>

<hr>

<h2 id="reuse">🧩 Re-using Components Programmatically</h2>
<p>
  The script is modular. Useful functions include:
</p>
<ul>
  <li><strong>I/O & preprocessing</strong>: <code>read_excel_header_row2</code>, <code>choose_sample_id</code>, <code>coerce_numeric</code>, <code>force_finite</code></li>
  <li><strong>Geometry & plotting</strong>: <code>_hex_centers</code>, <code>_hex_vertices</code>, <code>_draw_hex_map_packed</code>, <code>_add_cluster_boundaries</code>, <code>_label_cluster_components</code></li>
  <li><strong>Analytics</strong>: <code>compute_dense_umatrix</code>, <code>kmeans_scan_metrics_weighted</code></li>
  <li><strong>High-level plots</strong>: <code>plot_umatrix_hex_dense</code>, <code>plot_clusters_hex</code>, <code>plot_component_planes_hex</code>, <code>plot_codebook_pca_biplot</code>, <code>plot_box_by_cluster</code>, <code>plot_box_rows_by_scale</code>, <code>plot_cluster_strips_by_profile</code></li>
</ul>

<hr>

<h2 id="citation">📜 Citation & License</h2>
<p>
  Cite once the associated paper is published. Licensed under the <a href="LICENSE">MIT License</a>.
</p>

<p align="center">
  &copy; 2025 Matheus Rossi Santos — ORCID: 
  <a href="https://orcid.org/0000-0002-1604-381X">0000-0002-1604-381X</a>
</p>
