See [AGENTS.md](AGENTS.md) — the rules for working in this repository live there.

Current scientific state is in [PROJECT.md](PROJECT.md).

# Commit instructions
- every commit should be only at my name. NEVER add Claude among contributors. 
- Secrets. Never commit an API key, token, or password. If you do, deleting it in a later commit does not remove it — it's still in the history, and GitHub's automated scanners and other people's will find it. If it happens, assume the key is compromised, revoke it immediately, and regenerate.
- Large files. If a push fails with a size complaint, you've committed data. This is genuinely annoying to undo, which is why Step 1 comes first.
- Commit small and often. A commit per working function is about right. This matters more than it sounds: your commit history is visible on GitHub, and a repo with sixty thoughtful commits over two months reads very differently to a reviewer than one commit called "initial upload". It's evidence you actually built the thing incrementally.