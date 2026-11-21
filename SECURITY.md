# Security policy

## Reporting a vulnerability

Please report vulnerabilities privately through GitHub's private vulnerability reporting: open
the repository's **Security** tab and choose **Report a vulnerability**
(<https://github.com/agrissss/regimebot/security/advisories/new>). Do not open a public issue,
pull request or discussion for a security problem.

Include what you found, how to reproduce it, the commit or version you tested and the impact you
expect. You will get an acknowledgement, and the fix and its disclosure are coordinated with you
in the advisory.

## Supported versions

Only the `main` branch is supported. Fixes land there; there are no maintained release branches.

## Scope

regimebot is a research tool: backtests on stored public market data and paper trading on live
public data with simulated fills.

- It places no live orders. No code path can send an order to an exchange.
- It handles no exchange API keys or account credentials. Every exchange call uses public
  market-data endpoints.

In scope: anything that lets untrusted input (market data from the exchange, configuration,
command-line arguments, stored database rows, the optional alert webhook) execute code, read or
write files outside the project's data and report directories, leak a secret such as
`DATABASE_URL` or `ALERT_WEBHOOK_URL`, or corrupt stored results without notice; and
vulnerabilities in the pinned dependencies (`requirements.lock`) as this project uses them.

Out of scope: trading losses or strategy performance, and findings that need an attacker who
already controls the machine, its environment variables or its configuration file.
