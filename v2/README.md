![Reckoning](https://github.com/rf-peixoto/reckoning/blob/master/img/logo-v2.png)

[![rf-peixoto - Reckoning](https://img.shields.io/static/v1?label=rf-peixoto&message=Reckoning&color=red&logo=github)](https://github.com/rf-peixoto/reckoning)
[![stars - Reckoning](https://img.shields.io/github/stars/rf-peixoto/Reckoning?style=social)](https://github.com/rf-peixoto/reckoning)
[![forks - Reckoning](https://img.shields.io/github/forks/rf-peixoto/Reckoning?style=social)](https://github.com/rf-peixoto/reckoning)

**Reckoning** is a local web-based orchestrator for chaining CLI tools into repeatable workflows — built for recon, red team ops, and threat intelligence, but usable for any command-line tooling.

---

### Quick Start

```
git clone https://github.com/rf-peixoto/reckoning
cd reckoning
./install.sh
./start.sh
```

Then open `http://localhost:5000` in your browser. On Windows, use `install.bat` / `start.bat`.

---

### How It Works

Workflows are ordered chains of CLI tools. Each tool receives input from a previous step and passes its output to the next. A simple placeholder system handles the data flow:

| Placeholder | Value |
|---|---|
| `{domain}` or `{0}` | The target you entered at run time |
| `{1}`, `{2}`, `{3}`… | Output of step 1, 2, 3… |
| `{out_TOOL_ID}` | Stable reference to a specific tool's output |
| `{entry_name}` | Any value from your String Library |

---

### Setting Up Tools

Go to **Settings → Tool Library** and add your tools once. For each tool, define:

- **Path** — binary path or command name (e.g. `/usr/bin/nuclei`)
- **Default Args** — argument template with placeholders (e.g. `-l {subdomains} -o {output_file}`)
- **Update Args** — what to append when updating (e.g. `-update`)

When building a workflow, select a library tool as the base and override any field per-step if needed.

---

### String Library

Go to **Settings → String Library** to define named reusable values. Works for wordlists, API keys, tokens, hostnames, or any string you'd otherwise type repeatedly:

```
passwords  →  /usr/share/seclists/Passwords/rockyou.txt
shodan_key →  your_api_key_here
```

Reference them in any arguments template as `{passwords}`, `{shodan_key}`, etc.

---

### Creating Workflows

Workflows support four execution modes:

- **Once** — runs a single time (default)
- **Repeat N times** — runs N times with a configurable sleep between each run
- **Recurring** — runs every N minutes indefinitely until cancelled
- **Scheduled** — fires once at a specific date and time

Each tool step can have its own timeout, overriding the global setting in cases where one tool is expected to run much longer or shorter than others.

---

### Running Workflows

Click **Run** on the Dashboard. You can:

- Enter **multiple targets** (one per line) — each gets its own parallel execution
- Add **optional notes** to label the run
- Choose to **start immediately** or **schedule for later**

The Execution Detail page shows a live output feed while tools run, per-tool stdout/stderr, and a full event trail.

---

### Comparing Results

On the **Executions** page, check two completed executions from the **same workflow and target** and click **Compare Selected** to see a unified diff of each tool's output. Useful for tracking what changed between scans.

---

### Data & Backups

Everything is stored in `reckoning.db` (SQLite). Use **Settings → Data Management** to:

- **Download a backup** — a safe binary copy of the live database
- **Restore a backup** — upload a `.db` file (restart required to take full effect)

Execution history can be auto-pruned after N days via the **Execution History Retention** setting.

---

### Security

This tool runs shell commands on the machine it's installed on. **Do not expose it to the network.** Run it locally or on a dedicated isolated host. Only point it at targets you are authorized to test.

---

**Reckoning** is sponsored by _the-street.xyz_ — a private forum for cybersecurity, threat intelligence, and cipherpunk culture.

```
Bitcoin [BTC]   bc1qflkwmca89fzw9gfpg6dkf0es9ctnm2ktcq03dp
```
