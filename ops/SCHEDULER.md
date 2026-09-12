# Scheduling the paper loop on the Mac — what actually blocks it

**Status 2026-09-12: the loop is NOT running unattended.** Both schedulers fire
and both are denied by macOS TCC before the script starts.

## Evidence

`/var/mail/jash`, one entry per cron firing since install:

```
Thu 10 Sep 21:00  /bin/sh: .../logs/paper/cron.out: Operation not permitted
Fri 11 Sep 07:30  /bin/sh: .../logs/paper/cron.out: Operation not permitted
Fri 11 Sep 21:00  /bin/sh: .../logs/paper/cron.out: Operation not permitted
Sat 12 Sep 07:30  /bin/sh: .../logs/paper/cron.out: Operation not permitted
```

The machine never slept (`pmset -g`: sleep 0). Cron ran every time. The shell
it spawned could not open a file under `~/storage` — that is Transparency,
Consent & Control: the `cron` daemon has no Full Disk Access, so nothing it
launches can read or write the repo. launchd's user agent hit the same wall
earlier as `EX_CONFIG` from xpcproxy, which refuses to exec a script from a
path it cannot read. The identical command succeeds from a Terminal session
because Terminal holds the grant.

## The fix (one GUI action, cannot be scripted)

System Settings → Privacy & Security → **Full Disk Access** → `+` →
press `⌘⇧G` and enter `/usr/sbin/cron` → Add → toggle on.

Then verify without waiting for the schedule:

```
# next cron minute will write this if the grant took
( crontab -l; echo "* * * * * /bin/bash $PWD/scripts/paper_daily.sh >> $PWD/logs/paper/cron.out 2>&1 # PROBE" ) | crontab -
sleep 90; tail -3 logs/paper/cron.out          # expect a status line, not silence
crontab -l | grep -v "# PROBE" | crontab -     # remove the probe
```

If you would rather not grant cron FDA, granting it to `/bin/bash` also works,
and so does re-enabling the launchd agent (`ops/com.jash.tradingbot.paper.plist`)
after granting FDA to `/bin/bash` — launchd has the advantage of firing missed
runs on wake, which cron does not.

## Until then

Every session must be recorded by a run started from a Terminal session.
Missed sessions are recovered by the next run — the recorder writes one line
per session in the gap, each from that run's own replay — so a late run is
valid, but the gap is a gap in *unattended* operation and P2 counts more than
5 consecutive unrecoverable sessions as an abandonment criterion.
