"""Render a report from audited results; no search."""
import json
from pathlib import Path
HERE=Path(__file__).resolve().parent
read=lambda name:json.loads((HERE/name).read_text())
effects=read('effects.json');accounts=read('accounting.json');decision=read('decision.json')
lines=['# Component enumeration and cross-family probe','',
'The affine component minima are established, but the bounded cross-family probe does not demonstrate broader gains. Keep the component method as a specialized option. This experiment tests a particular detector, proposal set and dimension cutoff; it does not rule out other structural approaches.','',
'The [frozen plan](../../strategy_prototypes/component_search/PLAN.md) selected six large inputs using size and construction variety before search: two affine siblings, cyclic bicycle, balanced product, pair-partition lifted product, and local surface constructions. All methods share matrix-derived initialization and receive the same per-code CPU budgets. References and case labels never enter search. Construction blocks enter only the metadata method.','',
'## Exhaustive affine results','',
'Eight dimension-19 components were exhausted, visiting every one of their 524,287 nonzero kernel combinations. Two dimension-20 components were also exhausted. Their logical pairing ranks and nontrivial traversal counts passed an independent audit.','',
'| Input | Sector | Component size | Dimension | Minimum nontrivial weight | Copies |','|---|---|---:|---:|---:|---:|']
sessions=read('exact-affine/audit-components.json')['sessions'];groups={}
for s in sessions:
    key=s['case'],s['side'],len(s['coordinates']),s['dimension'],s['stats']['best']
    groups[key]=groups.get(key,0)+1
for key,count in sorted(groups.items()):lines.append('| '+' | '.join(map(str,(*key,count)))+' |')
lines += ['',
'For 684-12-73, the enumerated X components have minimum 51 and the Z components minimum 54. Other dimension-40 blocks were skipped, so this does not certify the minimum across all blocks or the full code. For 684-8-85, every component of both coordinate halves was exhausted: the minimum across single-block logical operators is exactly 54. A logical spanning both halves could still be lighter.','',
'Exhausting the selected spaces took about 58 ms and 61 ms per code, respectively, including preparation and initialization (validation excluded). No new lower full-code upper bound was found by this exact check. These results show that spending more time inside the enumerated spaces cannot improve their minima.','',
'## Fixed-budget comparison','',
'Entries list three independent seed outcomes. Lower is better. Each budget is per code on one CPU, half for X and half for Z; setup and export are included.','',
'| Input | Method | 2 seconds | 10 seconds |','|---|---|---|---|']
for c in effects:
    if c['grid']!='screen-2s':continue
    other=next(x for x in effects if x['grid']=='confirmation-10s' and x['case']==c['case'] and x['method']==c['method'])
    lines.append(f"| {c['case']} | {c['method']} | {c['weights']} | {other['weights']} |")
lines += ['',
'The metadata variant reproduces the established affine gains. The automatic detector misses those useful partitions. Both component variants reserve half of remaining time for guided search and release unused component time when spaces finish or none are eligible. Consequently a final weight can come from initialization or fallback; [effects.json](effects.json) reports component, initialization and fallback minima separately.','',
'## Coverage explains the limitations','',
'| Input | What the matrix audit found |','|---|---|',
'| Affine siblings | Documented halves split into dimension-19 components; automatic detector fails to recover these partitions. |',
'| Cyclic 682-182-76 | Detector recovers a partition, but the useful half kernels have dimension 91 and exceed the fixed 64 cap. |',
'| Balanced product 700-140-22 | Documented block pairs expose kernels of dimensions 4, 7, 16 and 19. Exhaustive minima across these retained spaces are X=48 and Z=56, above the guided result 22. Other dimension-140 spaces exceed the cap; some contain the known weight-22 reference. |',
'| Lifted product 664-170-18 | Individual documented 83-coordinate CPM blocks contain no nontrivial logicals. Multi-block combinations were not proposed. |',
'| Local surface 676-4-13 | Useful graph components have dimension 85, exceeding the fixed cap. |','',
'On the balanced product, the final weight 22 comes from initialization/fallback, not the component stage. Exhausting the retained small components shows that more compute in those same spaces cannot recover 22.\n\nThe dimension cap is an implementation/search-policy choice, not a mathematical impossibility. In particular, three unrelated inputs do not exercise component search under this configuration. The result therefore establishes no broad improvement, rather than establishing that structural search fails on those families. The independent [geometry audit](geometry-diagnostic.json) examines all documented proposals without the timed setup cutoff, distinguishing cutoff effects from missing logicals and excessive dimension.','',
'## Relabel diagnostic','',
'Each input was independently shuffled in both check rows and qubit columns. This is a one-seed diagnostic at two seconds, not a statistical robustness study. Metadata was not supplied to the automatic method. All permutations, transformed matrices and remapped references were checked.','',
'| Permuted input | Guided | Automatic components |','|---|---|---|']
for c in effects:
    if c['grid']=='permuted-2s' and c['method']=='guided':
        other=next(x for x in effects if x['grid']==c['grid'] and x['case']==c['case'] and x['method']=='auto-components')
        lines.append(f"| {c['case']} | {c['weights']} | {other['weights']} |")
lines += ['',
'Graph connectivity is invariant under these relabelings, but the reused partition detector scores adjacent check rows and is not row-order invariant. This study adds component splitting; it does not introduce a general graph-partition recovery algorithm.','',
'## Decision','',
f"Predeclared continuation criterion met: **{decision['continuation_criterion_met']}**. It required a component method on an unrelated input to beat every guided 10-second outcome by at least 10% in all three 2-second seeds, with component-stage evidence. Automatic and metadata outcomes are evaluated separately. [decision.json](decision.json) records the full result.",'',
'Keep the affine specialization and stop broad expansion of this configuration. A future effort would need a concrete new way to find useful partitions or handle logicals spanning proposed blocks; more runs of the same small-component policy will not address those gaps. The fixed cap also leaves larger restricted kernels untested. Cluster search and collision tuning remain second tier. No GPU work was performed.','',
'## Validation and artifacts','',
f"Completed {sum(a['configurations'] for a in accounts)} configurations with {sum(a['allocated_search_seconds'] for a in accounts):g} allocated search seconds. All {sum(a['saved_documents'] for a in accounts):,} saved witness documents passed their audits. Twenty-seven tests passed (nine new component tests plus eighteen unchanged native-engine tests); lint and the 27-file trusted verifier integrity check passed. All witness supports were checked for commutation and nontriviality through trusted algebra and persisted through the research kit. No full-code distance certificate or submission-quality gate is claimed.",'',
'The [final audit](final-audit.json) checks all current measured/supplemental sources and binaries, common initialization signatures across grids, portable witnesses, permutations, and unchanged artifacts from both preceding studies. [Accounting](accounting.json) separates allocated and actual search time from validation. [Witness index](witness-index.json) points to portable copies of the best per-sector documents, including relabeled cases. Raw logs, source archives, matrices and per-run counters remain in the four grid directories. [Reproduction instructions](REPRODUCE.md), hardware/package records, test/lint logs and SHA256SUMS accompany the report. No verifier, leaderboard, CI or publication changes were made.','']
(HERE/'README.md').write_text('\n'.join(lines))
