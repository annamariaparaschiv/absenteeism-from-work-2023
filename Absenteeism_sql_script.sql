CREATE DATABASE PredictedOutputs
USE PredictedOutputs

CREATE TABLE PredictedOutputs
(
    Reason_1 BIT NOT NULL,
	Reason_2 BIT NOT NULL,
	Reason_3 BIT NOT NULL,
	Reason_4 BIT NOT NULL,
    Month_Value INT NOT NULL,
    Transportation_Expense INT NOT NULL,
	Age INT NOT NULL,
	Body_Mass_Index INT NOT NULL,
	Education BIT NOT NULL,
	Children INT NOT NULL,
	Pets INT NOT NULL,
	Probability FLOAT NOT NULL,
	Prediction BIT NOT NULL
)

SELECT * FROM PredictedOutputs