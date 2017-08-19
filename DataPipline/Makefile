AllTest:AllModel
	python AllStockTestModel.py
AllModel:DowTest
	python AllStockPredictionModelling.py
DowTest:DowModel
	python DowPredictionEmail.py
DowModel:DataIngestion
	python DowPredictionJob.py
DataIngestion:
	python DataIngestion.py