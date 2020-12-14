from unittest.mock import MagicMock
import sys
sys.modules["xgboost"] = MagicMock()

from explainerdashboard import RegressionExplainer, ExplainerDashboard

explainer = RegressionExplainer.from_file("explainer.joblib")
# you can override params during load from_config:
db = ExplainerDashboard.from_config(explainer, "dashboard.yaml", title="Test")

app = db.flask_server()

# run waitress-serve --port=8070 dashboard:app in command line
