![Reckoning](https://github.com/rf-peixoto/reckoning/blob/master/img/logo-v2.png)

[![rf-peixoto - Reckoning](https://img.shields.io/static/v1?label=rf-peixoto&message=Reckoning&color=red&logo=github)](https://github.com/rf-peixoto/reckoning)
[![stars - Reckoning](https://img.shields.io/github/stars/rf-peixoto/Reckoning?style=social)](https://github.com/rf-peixoto/reckoning)
[![forks - Reckoning](https://img.shields.io/github/forks/rf-peixoto/Reckoning?style=social)](https://github.com/rf-peixoto/reckoning)

**Reckoning** is an orchestrator designed for offensive operations, reconnaissance, and monitoring. However, its modular nature allows you to create workflows for any and all types of tools accessible via the command line. Naturally, its development focuses on facilitating the creation of Red Team, Penetration Testing, and Threat Intelligence operations. Share your workflow configurations, tools, wordlists, etc. with the community.

### tl;dr
1. **Clone this repo** - `git clone https://github.com/rf-peixoto/reckoning`
2. **Install** - `install.sh`
3. **Start** - `start.sh`
4. **Point your browser** - `http://localhost:5000`

---

### Installing Tools:
- Go to the *Settings* page.
- Define the name, path, arguments, and update command.
- **Reckoning** uses a string substitution system accessible to all tools in a running workflow. `{0}` is equivalent to the initial domain specified at the start of execution. The outputs of each tool can be accessed by their number in the workflow order: `{1}`, `{2}`, `{3}`...


### Defining Wordlists:
- Go to the *Settings* page.
- To avoid having to type `/usr/share/seclists...` for every command, you can define names for wordlists by choosing their name and path. This way, to call the wordlist in a command, simply type `{name}`.
- You can use this mechanism to define API keys. It's a simple string substitution; just set the PATH value as your key or token.

### Creating Workflows:
- **Reckoning** currently has three workflow execution modes:
  - *Run Once (default)*: Executes only once.
  - *Recurring*: Runs continuously every _N_ minutes.
  - *Run N Times*: executes at intervals of _N_ minutes until _N(2)_ executions are completed.
- Choose which tools to use and in what order. You can completely change the options or use the pre-configured base on the _Settings_ page. You can also set a color for organizational purposes.
- Follow the progress on the _Executions_ page.


---
**Reckoning** is sponsored by:  _the-street.xyz_, a private forum for cybersecurity, threat intelligence, and cipherpunk culture. Feel free to support us too:
```
Bitcoin   [BTC]     bc1qflkwmca89fzw9gfpg6dkf0es9ctnm2ktcq03dp
```
