Do not forget to install ADBKeyboard.

By accident I doubled the limit of max steps as I assumed that Android World would give me the expected numbers of steps but they gave me double that.
If there is an error with Gemini (e.g., server problems, no internet connection), a task is restarted.

I suspect I do not have the correct version of the emulator? Maybe an incorrect image?

Additionally, I graded some task per hand because of  grading errors like this:
android_world/agents/task/0/fbdc5d7ea9af17de
The agent clearly followed the instructions but did not get a success. I changed the grading.

ContactsNewContactDraft: the autograding did not give success, but when I used the human agent it was a success? strange. I changed grading to succes.
MarkorMergeNotes: the autograding said no succes, but the agent did exactly as specified (Maybe some kind of bug?)

I do no know if this has anything to do with the fact that the accessibility forwarder keeps crashing or warning like these:
W0514 12:18:03.743592 126966152508928 task_eval.py:123] Skipping app snapshot loading : Snapshot not found in /data/data/android_world/snapshots/net.gsantner.markor.

OpenAppTaskEval task does not seem to be loading properly. It was graded as success (by me and the autograding).
Similarly for the clock task (ClockStopWatchPausedVerify, ClockStopWatchRunning, ClockTimerEntry)
For the last 2 the agent did what was expected, but the automatic grading said failure. I graded those as success.

So far you could just remove 6 success and 6 task to calculate the success rate?
No, I will just subtract those 6 from the success? To get a lower boundary? Yes, I will probably do that so do not wonder if there is a difference of 6.


While testing I saw that these Simple apps have a different name. I thought it was kinda pointless to watch the agent fail for hours trying to find the apps, so I added a tip.
Maybe the name of the Simple Apps changed? At least I think the tasks are wrong these are not the Pro version. Those should cost money.
Additionally I added knowledge about which SMS app to use. "Simple SMS Messenger" -> "SMS Messenger" Their naming scheme is not even consistent.