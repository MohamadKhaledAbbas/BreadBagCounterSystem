# BreadBag System Codebase Workflow

This guide documents the standard workflow to **sync your project codebase** and **convert models** for deployment.

---

## 1. Sync Your Codebase from Windows to RDK Board

Use `rsync` to synchronize your project with custom include/exclude rules using a filter file.

```bash
rsync -avz --progress \
    --filter="merge rsync.rules" \
    /mnt/c/Users/Khaled/PyCharmMiscProject/ \
    sunrise@rdkboard:/home/sunrise/BreadCounting/
```

- `-a` : Archive mode (recursive copy, preserves permissions)
- `-v` : Verbose output
- `-z` : Compress file data during transfer
- `--progress` : Show progress during transfer
- `--filter="merge rsync.rules"` : Apply filter rules (see [`rsync.rules`](#rsyncrules-example))
- `source` : `/mnt/c/Users/Khaled/PyCharmMiscProject/`
- `destination` : `sunrise@rdkboard:/home/sunrise/BreadCounting/`

---

## 2. Start Model Conversion Environment (Docker)

Launch a Docker container with the OpenExplorer AI toolchain, mounting your model directory for conversion tasks.

```bash
docker run -it --rm \
    -v "C:\Users\Khaled\PyCharmMiscProject\data:/data" \
    openexplorer/ai_toolchain_ubuntu_20_x5_cpu:v1.2.8 \
    /bin/bash
```

- `-it` : Interactive terminal
- `--rm` : Remove the container when it exits
- `-v ...` : Mount local directory to `/data` inside the container
- `openexplorer/ai_toolchain_ubuntu_20_x5_cpu:v1.2.8` : Docker image containing model tools

---

## 3. Convert ONNX Model to BIN Format

Run the model converter script inside the Docker container:

```bash
python3 model_converter/mapper.py \
    --onnx model/best_classify.onnx \
    --cal-images model_converter/Classify_Calibration
```

- `model_converter/mapper.py` : Conversion script
- `--onnx model/best_classify.onnx` : Input ONNX model file
- `--cal-images model_converter/Classify_Calibration` : Calibration images directory

---

## rsync.rules Example

Example content for your `rsync.rules` filter file:

```text
+ src/
+ src/***
+ data/
+ data/classes/
+ data/classes/***
+ data/model/
+ data/model/***
+ data/db/
+ data/db/***
+ .gitignore
+ config.py
+ db_cli.py
+ main.py
+ main_ui.py
+ run_app.sh
+ stop_app.sh
- *
```
This filter includes only the critical source, data, and script files for syncing, and excludes everything else.

---

## Workflow Summary

| **Step**           | **Command** | **Purpose**                               |
|--------------------|-------------|-------------------------------------------|
| Sync codebase      | rsync       | Efficient, selective project sync         |
| Launch Docker env  | docker      | Isolated Python/AI toolchain for models   |
| Model conversion   | python3     | Convert .onnx model to .bin format        |

---

**Tip:** Place these commands in your README or project wiki for quick reference!