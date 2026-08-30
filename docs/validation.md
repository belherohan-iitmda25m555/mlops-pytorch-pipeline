# Validation Evidence

## Unit Tests

Command:

```powershell
python -m pytest -q
```

Result:

```text
2 passed
```

## Dockerfile Validation

Commands:

```powershell
docker build --check -f docker/Dockerfile.train .
docker build --check -f docker/Dockerfile.serve .
```

Result for both Dockerfiles:

```text
Check complete, no warnings found.
```

## Containerized Training

Command:

```powershell
docker run --rm -v "${PWD}/data:/app/data" -v "${PWD}/checkpoints:/app/checkpoints" mlops-train:v1
```

Result:

```json
{"event":"training_started","device":"cpu","config_path":"/app/configs/training_config.yaml"}
{"event":"epoch_completed","epoch":1,"training_loss":0.469121,"validation_loss":0.327382,"validation_accuracy":0.8814}
{"event":"checkpoint_saved","path":"checkpoints/classifier_v1.pt"}
{"event":"epoch_completed","epoch":2,"training_loss":0.307934,"validation_loss":0.275261,"validation_accuracy":0.8994}
{"event":"checkpoint_saved","path":"checkpoints/classifier_v1.pt"}
{"event":"training_finished","best_validation_loss":0.275261}
```

## Containerized API

Health response:

```json
{"status":"healthy","model_path":"/app/checkpoints/classifier_v1.pt"}
```

Prediction response:

```json
{
  "predicted_class": "Bag",
  "predicted_class_index": 8,
  "probabilities": {
    "T-shirt/top": 0.118797,
    "Trouser": 0.096517,
    "Pullover": 0.051359,
    "Dress": 0.066031,
    "Coat": 0.049191,
    "Sandal": 0.058973,
    "Shirt": 0.095305,
    "Sneaker": 0.036469,
    "Bag": 0.351484,
    "Ankle boot": 0.075874
  }
}
```

Both containerized API requests returned HTTP 200.

## Corrected Kubernetes Manifest Validation

Command:

```powershell
kubectl apply --dry-run=client -f k8s\
```

Result:

```text
configmap/training-config created (dry run)
horizontalpodautoscaler.autoscaling/model-serving created (dry run)
namespace/ml-training created (dry run)
persistentvolumeclaim/training-data-pvc created (dry run)
persistentvolumeclaim/model-checkpoints-pvc created (dry run)
deployment.apps/model-serving created (dry run)
service/model-serving created (dry run)
job.batch/fashion-mnist-training created (dry run)
```

All eight Kubernetes resources passed client-side schema and YAML validation.

## Kubernetes Training Resources

Command:

```powershell
kubectl get jobs,pods,pvc -n ml-training
```

Result after training:

```text
NAME                               STATUS     COMPLETIONS   DURATION
job.batch/fashion-mnist-training   Complete   1/1           2m34s

NAME                               READY   STATUS      RESTARTS
pod/fashion-mnist-training-xgzkb   0/1     Completed   0

NAME                                          STATUS   CAPACITY   ACCESS MODES
persistentvolumeclaim/model-checkpoints-pvc   Bound    1Gi        RWO
persistentvolumeclaim/training-data-pvc       Bound    2Gi        RWO
```

The dataset and trained checkpoint were stored on separate persistent volumes as required.

## Kubernetes Training Log

```json
{"event":"training_started","device":"cpu","config_path":"/app/configs/training_config.yaml"}
{"event":"epoch_completed","epoch":1,"training_loss":0.469121,"validation_loss":0.327382,"validation_accuracy":0.8814}
{"event":"checkpoint_saved","path":"/app/checkpoints/classifier_v1.pt"}
{"event":"epoch_completed","epoch":2,"training_loss":0.307934,"validation_loss":0.275261,"validation_accuracy":0.8994}
{"event":"checkpoint_saved","path":"/app/checkpoints/classifier_v1.pt"}
{"event":"training_finished","best_validation_loss":0.275261}
```

## Kubernetes Serving Resources

Command:

```powershell
kubectl get deployments,pods,services,hpa -n ml-training
```

Result:

```text
NAME                            READY   UP-TO-DATE   AVAILABLE
deployment.apps/model-serving   2/2     2            2

NAME                                READY   STATUS      RESTARTS
pod/fashion-mnist-training-xgzkb    0/1     Completed   0
pod/model-serving-d9975f6cc-2lk7h   1/1     Running     0
pod/model-serving-d9975f6cc-qrtkx   1/1     Running     0

NAME                    TYPE        PORT(S)
service/model-serving   ClusterIP   80/TCP
```

The serving Deployment runs two healthy replicas using the checkpoint produced by the training Job.

## Horizontal Pod Autoscaler

Command:

```powershell
kubectl get hpa -n ml-training
```

Result:

```text
NAME            REFERENCE                  TARGETS      MINPODS   MAXPODS   REPLICAS
model-serving   Deployment/model-serving   cpu: 0%/70%  2         4         2
```

The HPA successfully received CPU metrics and retained the required two minimum replicas.

## Kubernetes API Validation

Port-forward command:

```powershell
kubectl port-forward -n ml-training service/model-serving 8080:80
```

Health command:

```powershell
curl.exe http://127.0.0.1:8080/health
```

Health response:

```json
{"status":"healthy","model_path":"/app/checkpoints/classifier_v1.pt"}
```

Prediction command:

```powershell
curl.exe -X POST http://127.0.0.1:8080/predict -F "image=@data/test_image.png"
```

Prediction response:

```json
{
  "predicted_class": "Bag",
  "predicted_class_index": 8,
  "probabilities": {
    "T-shirt/top": 0.118797,
    "Trouser": 0.096517,
    "Pullover": 0.051359,
    "Dress": 0.066031,
    "Coat": 0.049191,
    "Sandal": 0.058973,
    "Shirt": 0.095305,
    "Sneaker": 0.036469,
    "Bag": 0.351484,
    "Ankle boot": 0.075874
  }
}
```

Both Kubernetes API requests completed successfully. This confirms that the Service routed traffic to the two-replica serving Deployment and that the API loaded the checkpoint created by the Kubernetes training Job.