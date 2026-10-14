# Code provenance and independence

Preserve `Qu-ANTI-zation/` without modification or deletion. Installation,
execution and ordinary tests must work without that directory.

Before porting code, record the upstream URL and full commit SHA in
`docs/provenance.md`. Preserve original copyright and license notices and keep
license texts in `licenses/`. Derived file headers should identify the original
project, file path, license and provenance document. Add function-level attribution
only when a file combines sources that its header cannot clearly identify.

Keep source mappings and significant changes in `docs/provenance.md`.
Record checkpoint sources and weight licensing separately from code licensing.
Do not copy third-party code without an identified license; use an independent
implementation of the formulas where necessary.
