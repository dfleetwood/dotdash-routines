"""
Linear Regression Routine Plugin

This plugin fits a simple linear regression model to data from a dataset.
"""

from typing import List, Optional, Dict, Any, Union

from dotdash.decorators import routine
from dotdash.models.routine_types import (
    DatasetOutput, Metric, Visualization, 
    NumericField, TextField, BooleanField, Result, DatasetReference
)
from dotdash.models.routine_types import NumericField, CategoricalField, BooleanField
from dotdash.utils.dataset_helpers import load_dataset


class RegressionResult(Result):
    """Results from linear regression analysis"""
    metrics: List[Metric]
    predictions: Optional[DatasetOutput] = None


@routine(
    name="Linear Regression",
    description="Fit a linear regression model to numeric data",
    path = 'Regression/Linear',
    version="0.1.0"
)
def linear_regression(
    data: DatasetReference,
    x_column: List[Union[NumericField, CategoricalField, BooleanField]],
    y_column: NumericField,
    predict_new_values: bool = True
) -> RegressionResult:
    """
    Fit a linear regression model to the specified columns.
    
    Args:
        dataset: Input dataset containing the data
        x_column: Column name for independent variable (X)
        y_column: Column name for dependent variable (Y)
        predict_new_values: Whether to generate predictions
        
    Returns:
        Regression results including model coefficients and statistics
    """

    from sklearn.linear_model import LinearRegression
    from sklearn.metrics import mean_squared_error, r2_score

    print ('print in data')
    print (data)

    # Load the dataset using the helper
    df = load_dataset(data).to_pandas()
    
    X = df[x_column].values
    y = df[y_column].values
    
    # Fit the linear regression model
    model = LinearRegression()
    model.fit(X, y)
    
    # Make predictions using the model
    y_pred = model.predict(X)
    
    # Calculate metrics
    mse = mean_squared_error(y, y_pred)
    r2 = r2_score(y, y_pred)
    
    # Create metrics
    metrics = [
        Metric(
            name="R²",
            value=float(r2),
            label="R-squared",
            description="Coefficient of determination"
        ),
        Metric(
            name="MSE",
            value=float(mse),
            label="Mean Squared Error",
            description="Average squared difference between predicted and actual values"
        )
    ]
    
    # Create output dataset with original data and predictions
    if predict_new_values:
        # Build the output dataset
        result_df = df.copy()
        result_df[f"{y_column}_predicted"] = y_pred
        result_df["residual"] = y - y_pred
        

        output_dataset = DatasetOutput(
            name=f"Linear Regression Results: {x_column} vs {y_column}",
            description=f"Predicted values for {y_column} based on {x_column}",
            data=result_df.to_dict(orient = 'records')
        )
    else:
        output_dataset = None
    
    # Return combined results
    return RegressionResult(
        metrics=metrics,
        predictions=output_dataset
    ) 