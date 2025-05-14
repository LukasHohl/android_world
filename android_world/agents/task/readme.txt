By accident I doubled the limit of max steps as I assumed that Android World would give me the expected numbers of steps but they gave me double that.

Additionally, I graded some task per hand because of  grading errors like this:
android_world/agents/task/0/fbdc5d7ea9af17de
The agent clearly followed the instructions but did not get a success. I changed that.

I do no know if this has anything to do with the fact that the accessibility forwarder keeps crashing or warning like these:
W0514 12:18:03.743592 126966152508928 task_eval.py:123] Skipping app snapshot loading : Snapshot not found in /data/data/android_world/snapshots/net.gsantner.markor.

OpenAppTaskEval task does not seem to be loading properly. It was graded as success.
Similarly for the clock task (ClockStopWatchPausedVerify, ClockStopWatchRunning, ClockTimerEntry)
For the last 2 the agent did what was expected, but the automatic grading said failure. I graded those as success.