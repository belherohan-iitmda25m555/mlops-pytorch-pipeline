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

Both requests returned HTTP 200.

## Kubernetes Manifest Validation

Command:

```powershell
kubectl apply --dry-run=client -f k8s\
```

Result:

```text
configmap/training-config created (dry run)
horizontalpodautoscaler.autoscaling/fashion-mnist-serving created (dry run)
namespace/mlops created (dry run)
persistentvolumeclaim/model-checkpoints-pvc created (dry run)
deployment.apps/fashion-mnist-serving created (dry run)
service/fashion-mnist-serving created (dry run)
job.batch/fashion-mnist-training created (dry run)
```

## Kubernetes Training

Training Job result:

```json
{"event":"epoch_completed","epoch":1,"training_loss":0.469121,"validation_loss":0.327382,"validation_accuracy":0.8814}
{"event":"checkpoint_saved","path":"checkpoints/classifier_v1.pt"}
{"event":"epoch_completed","epoch":2,"training_loss":0.307934,"validation_loss":0.275261,"validation_accuracy":0.8994}
{"event":"checkpoint_saved","path":"checkpoints/classifier_v1.pt"}
{"event":"training_finished","best_validation_loss":0.275261}
```

## Kubernetes Serving Resources

```text
NAME                                    READY   UP-TO-DATE   AVAILABLE
deployment.apps/fashion-mnist-serving   1/1     1            1

NAME                                         READY   STATUS      RESTARTS
pod/fashion-mnist-serving-78dcb64c46-6fdbz   1/1     Running     0
pod/fashion-mnist-training-r5t4z             0/1     Completed   0

NAME                            TYPE        PORT(S)
service/fashion-mnist-serving   ClusterIP   8080/TCP
```

## Horizontal Pod Autoscaler

```text
NAME                    TARGETS      MINPODS   MAXPODS   REPLICAS
fashion-mnist-serving   cpu: 2%/70%  1         3         1
```

## Kubernetes API Validation

The port-forwarded Kubernetes Service returned the same successful `/health` and `/predict` responses documented above, confirming that the API loaded the checkpoint created by the Kubernetes training Job.