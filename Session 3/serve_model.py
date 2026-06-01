import json
import pandas as pd
import mlflow
import socketserver
import http.server
import traceback

mlflow.set_tracking_uri("http://127.0.0.1:8080")
print("Loading model...")
model = mlflow.sklearn.load_model("models:/class-churn-model/latest")
print("Model loaded. Server starting...")

class Handler(http.server.BaseHTTPRequestHandler):
    def do_POST(self):
        try:
            data = json.loads(self.rfile.read(int(self.headers['Content-Length'])))
            df = pd.DataFrame(
                data['dataframe_split']['data'],
                columns=data['dataframe_split']['columns']
            )
            # Convert float columns to int only when all values are whole numbers
            for col in df.columns:
                if df[col].dtype == 'float64':
                    if (df[col] % 1 == 0).all():
                        df[col] = df[col].astype('int64')
            preds = model.predict(df)
            pred_list = [int(p) for p in preds]
            resp = json.dumps({"predictions": pred_list}).encode()
            self.send_response(200)
            self.send_header('Content-Type', 'application/json')
            self.end_headers()
            self.wfile.write(resp)
        except Exception as e:
            traceback.print_exc()
            self.send_response(500)
            self.end_headers()
            self.wfile.write(str(e).encode())

    def log_message(self, *_):
        pass

socketserver.TCPServer.allow_reuse_address = True
with socketserver.TCPServer(('127.0.0.1', 5000), Handler) as httpd:
    print("Server running at http://127.0.0.1:5000/invocations")
    httpd.serve_forever()
