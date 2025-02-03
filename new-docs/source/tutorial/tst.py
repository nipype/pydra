from pydra.tasks.testing import UnsafeDivisionWorkflow
from pydra.engine.submitter import Submitter

# This workflow will fail because we are trying to divide by 0
wf = UnsafeDivisionWorkflow(a=10, b=5, denominator=2)

if __name__ == "__main__":
    with Submitter(worker="cf") as sub:
        result = sub(wf)


# from pydra.tasks.testing import UnsafeDivisionWorkflow
# from pydra.engine.submitter import Submitter

# # This workflow will fail because we are trying to divide by 0
# failing_workflow = UnsafeDivisionWorkflow(a=10, b=5).split(denominator=[3, 2, 0])

# if __name__ == "__main__":
#     with Submitter(worker="cf") as sub:
#         result = sub(failing_workflow)

#     if result.errored:
#         print("Workflow failed with errors:\n" + str(result.errors))
#     else:
#         print("Workflow completed successfully :)")
