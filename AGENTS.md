<!-- LOVABLE:BEGIN -->
> [!IMPORTANT]
> This project is connected to [Lovable](https://lovable.dev). Avoid rewriting
> published git history — force pushing, or rebasing/amending/squashing commits
> that are already pushed — as it rewrites history on Lovable's side and the
> user will likely lose their project history.
>
> Commits you push to the connected branch sync back to Lovable and show up in
> the editor, so keep the branch in a working state.
<!-- LOVABLE:END -->
- Accounts use Lovable Cloud auth and scans live in the `scans` table with owner-only RLS; never store credentials or history in browser storage.
- Classification runs only in the authenticated `classifyWaste` server function, which also saves the row, so clients cannot forge results.
- Bin text and the uncertainty threshold are owned by `normalizePrediction` in scan-schema.ts; the Python service in `backend/` mirrors the same rules and is used when `PYTHON_CLASSIFIER_URL` is set.
- Keep the React interface in one route file.
