from __future__ import annotations
from sklearn.calibration import CalibratedClassifierCV
from sklearn.pipeline import Pipeline


def calibrate_pipeline(
    fitted_pipe: Pipeline,
    X_valid,
    y_valid,
    method: str = "isotonic",
) -> Pipeline:
    """
    Wrap the final estimator in a calibrated classifier while keeping preprocessing fixed.
    Assumes fitted_pipe = Pipeline([... ('clf', estimator)]) and already fitted.
    """
    if not isinstance(fitted_pipe, Pipeline):
        raise ValueError("fitted_pipe must be a sklearn Pipeline")

    steps = fitted_pipe.steps
    if not steps or steps[-1][0] != "clf":
        raise ValueError("Expected last pipeline step to be named 'clf'")

    preproc = Pipeline(steps=steps[:-1])
    clf = steps[-1][1]

    Xv = preproc.transform(X_valid)

    cal = CalibratedClassifierCV(base_estimator=clf, method=method, cv="prefit")
    cal.fit(Xv, y_valid)

    calibrated_pipe = Pipeline(steps=steps[:-1] + [("clf", cal)])
    return calibrated_pipe