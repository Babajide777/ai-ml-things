from sklearn.metrics import mean_absolute_error, root_mean_squared_error, r2_score


def regression_metrics(y_true, y_pred, target_unit_usd=1.0):
    """Errors in dollars after multiplying by target_unit_usd (1.0 if already $)."""
    return {
        "MAE ($)": mean_absolute_error(y_true, y_pred) * target_unit_usd,
        "RMSE ($)": root_mean_squared_error(y_true, y_pred) * target_unit_usd,
        "R2": r2_score(y_true, y_pred),
    }


def format_results(results):
    formatted = results.copy().astype(object)
    for col in ["MAE ($)", "RMSE ($)"]:
        formatted[col] = results[col].map(lambda v: f"${v:,.2f}")
    formatted["R2"] = results["R2"].map(lambda v: f"{v:.4f}")
    return formatted
