# Review before submission

This report is an AI-assisted draft based on the supplied repository snapshot, not a submission-ready paper. The group must read and understand the cited sources and verify all claims, results and final wording. No new experiments were run and no raw results were altered.

1. Replace the title-page placeholders with the actual group members, student numbers and submission date. Delete unused member placeholders rather than assuming four members.
2. The Muon project webpage could not be accessed during this review. Compare the report with the live project-specific brief.
3. Read each cited source and check the interpretation against the full paper. Confirm the group's actual use of AI before describing it; no history of who used AI was invented.
4. Audit numerical claims, including differences in percentage averaging conventions across figures and tables, against raw results and analysis code. Verify run counts and protocol chronology against the notebook and version history.
5. Tests and the smoke test were not run: this environment does not have the project dependencies, including PyTorch and pytest. Run the README setup and tests in the documented environment. Validate reproduction from a fresh checkout, including checkpoint regeneration.
6. Existing results were not rerun. E1 evaluation uses the same seeds as rate selection; evaluate selected rates on fresh seeds if resources permit. Consider a further experiment targeting substantially lower delta and/or direct equal-time tuning. Treat added work as exploratory unless independently planned before inspection.
7. Check captions, figure readability, bibliography and the final PDF after inserting author details. Compile again and ensure the final document including all figures and references remains at most 10 pages.
