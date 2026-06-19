# deploy_pipeline.py
#
# This creates a PREFECT DEPLOYMENT for the flow defined in
# pipeline_flow.py. A deployment is separate from the flow itself:
#   - the FLOW (pipeline_flow.py)  = what code runs
#   - the DEPLOYMENT (this file)   = when/how it runs, and that's
#                                    something you can turn on, off,
#                                    or change WITHOUT touching the
#                                    flow's code at all.
#
# Run it with:
#   python3 deploy_pipeline.py
#
# This does NOT run the pipeline immediately — it just registers the
# schedule with Prefect. The pipeline only fires automatically once
# you have a Prefect worker running to execute scheduled work (we'll
# cover that after you confirm the config below makes sense).

from pipeline_flow import training_pipeline

# ── CONFIGURATION — change these to control scheduling ─────────
# This is the "switch" you asked for. Everything below is plain,
# readable config — no need to touch the flow code to change behavior.

SCHEDULING_ENABLED = True     # <-- master on/off switch
CRON_SCHEDULE = "*/1 * * * *"    # <-- "every day at 6:00 AM" in cron syntax

# Cron syntax quick reference (5 fields: minute hour day month weekday):
#   "0 6 * * *"     -> every day at 6:00 AM
#   "0 */6 * * *"   -> every 6 hours
#   "0 6 * * 1"     -> every Monday at 6:00 AM
#   "*/15 * * * *"  -> every 15 minutes (useful for TESTING the schedule
#                      fires at all, without waiting a full day)


def build_deployment():
    if not SCHEDULING_ENABLED:
        print("Scheduling is DISABLED (SCHEDULING_ENABLED = False).")
        print("Deploying with NO schedule — this flow will only run when")
        print("triggered manually (python3 pipeline_flow.py, or via the")
        print("Prefect UI's 'Run' button).")
        training_pipeline.serve(
            name="fire-risk-pipeline-manual",
        )
    else:
        print(f"Scheduling is ENABLED. Cron: {CRON_SCHEDULE}")
        training_pipeline.serve(
            name="fire-risk-pipeline-scheduled",
            cron=CRON_SCHEDULE,
        )


if __name__ == "__main__":
    build_deployment()