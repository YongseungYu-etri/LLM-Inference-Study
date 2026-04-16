---
title: "Private — Onboarding Doc Access"
type: "docs"
bookHidden: true
bookToc: false
---

# Private — Onboarding Doc Access

This page provides download access to the **Claude session onboarding document**
for this project. The content is AES-256-CBC encrypted with a shared password.

Access is intentionally not linked from the main navigation (`bookHidden: true`),
but the encrypted file URL is public. The file itself is useless without the password.

## 1. Download

| File | Path |
|------|------|
| Encrypted onboarding doc | [`ONBOARDING.md.enc`](/LLM-Inference-Study/private/ONBOARDING.md.enc) |

Direct URL:
```
https://yongseungyu-etri.github.io/LLM-Inference-Study/private/ONBOARDING.md.enc
```

## 2. Decrypt (local machine)

```bash
# One-liner: download + decrypt with the password
curl -sL https://yongseungyu-etri.github.io/LLM-Inference-Study/private/ONBOARDING.md.enc \
  -o ONBOARDING.md.enc

openssl enc -d -aes-256-cbc -pbkdf2 -iter 100000 \
  -in ONBOARDING.md.enc \
  -out ONBOARDING.md \
  -k "<PASSWORD>"
```

Replace `<PASSWORD>` with the shared password (not published here).

## 3. Use

The decrypted `ONBOARDING.md` is a self-contained briefing document
that a new Claude session can read to quickly understand:
- Project goal and structure
- User profile and preferences
- Hardware / software baseline
- Study methodology conventions
- Recent decisions and corrections (e.g., cuBLAS → cuBLASLt)
- How to add new subsections
- How to build and deploy

Use it by pasting its contents (or the relevant portions) into a new Claude
session at the start of the conversation, or by attaching it as a file.

## 4. Re-encryption (if editing)

If you need to update the onboarding doc:

```bash
# 1. Decrypt → edit → re-encrypt
openssl enc -d -aes-256-cbc -pbkdf2 -iter 100000 \
  -in static/private/ONBOARDING.md.enc -out ONBOARDING.md -k "<PASSWORD>"

# ... edit ONBOARDING.md ...

openssl enc -aes-256-cbc -salt -pbkdf2 -iter 100000 \
  -in ONBOARDING.md -out static/private/ONBOARDING.md.enc -k "<PASSWORD>"

# 2. Commit + push
git add static/private/ONBOARDING.md.enc
git commit -m "Update onboarding doc"
git push
```

---

## Threat Model (transparency)

- **Repo visibility**: public. Anyone can clone.
- **Encrypted file**: AES-256-CBC with 100000-iteration PBKDF2. Practically unbreakable without the password.
- **Password**: shared out-of-band (not in repo, not in commits, not on the site).
- **URL**: technically indexable by GitHub/search engines, but the file content is encrypted noise.
- **Caveat**: this is "privacy via encryption" not true access control. If the password leaks, the file leaks.

This level of protection is appropriate for: onboarding context, project notes, non-sensitive preferences.
**Do NOT use this mechanism for**: secrets, credentials, PII, or commercial IP.
