# U-Net Projekt Struktur - Komplette Anleitung

## 📋 Überblick: Was du alles machen musst

### Reihenfolge der Schritte:
1. **Daten vorbereiten** (image_snipper.py) - EINMALIG
2. **Dataloader erstellen** (dataloader.py) - Funktionen schreiben
3. **Model definieren** (U_net.py) - Fehler beheben
4. **Training durchführen** (train.py) - Alles zusammenfügen
5. **Evaluieren & Testen** - Ergebnisse anschauen

---

## 🔧 SCHRITT 1: Daten Vorbereitung (image_snipper.py)

### Status: ✅ Funktioniert bereits
### Was passiert hier:
- Große Satellitenbilder werden in kleine 512x512 Patches zerschnitten
- Bilder werden normalisiert (wichtig für Training!)
- Alles wird als .npy Dateien gespeichert

### Input:
```
data/aerial/training_data/
├── bild1_image.tif    (z.B. 2000x2000, 4 Kanäle: NIR, R, G, B)
├── bild1_mask.tif     (z.B. 2000x2000, 1 Kanal: Klassen 0/1)
├── bild2_image.tif
└── bild2_mask.tif
```

### Output:
```
data/aerial/training_data/
├── img_tiles/
│   ├── img_0001.npy   (512, 512, 4) - normalisiert
│   ├── img_0002.npy
│   └── ...
└── mask_tiles/
    ├── mask_0001.npy  (512, 512, 1) - Klassen
    ├── mask_0002.npy
    └── ...
```

### Ausführung:
```bash
cd a:\STUDIUM\05_Herbstsemester25\PA2\src\U-net
python image_snipper.py
```

### Kontrolle:
- Überprüfe, dass img_tiles/ und mask_tiles/ Ordner existieren
- Zähle Dateien: `len(os.listdir("img_tiles"))`
- Lade eine Datei: `img = np.load("img_tiles/img_0001.npy")`
- Prüfe Shape: `img.shape` sollte `(512, 512, 4)` sein

---

## 📦 SCHRITT 2: Dataloader (dataloader.py)

### Status: ⚠️ MUSS NOCH GEMACHT WERDEN
### Was fehlt:
Du brauchst Funktionen um die .npy Dateien zu laden und ein TensorFlow Dataset zu erstellen.

### Fehlende Funktionen:

#### Funktion 1: load_npy_dataset()
```python
def load_npy_dataset(img_path, mask_path):
    """
    Input:  img_path = "path/to/img_tiles"
            mask_path = "path/to/mask_tiles"
    Output: dataset = tf.data.Dataset mit (image, mask) Paaren
    
    Beispiel:
    - Liste alle .npy Dateien in img_path
    - Sortiere sie (wichtig: img_0001.npy passt zu mask_0001.npy)
    - Lade sie mit np.load()
    - Erstelle Dataset mit tf.data.Dataset.from_generator() oder .from_tensor_slices()
    """
```

**Datenformat im Dataset:**
- Jedes Element: `(image, mask)` Tupel
- image: Shape `(512, 512, 4)`, dtype=`float32`, Werte sind normalisiert
- mask: Shape `(512, 512, 1)`, dtype=`float32` oder `int32`, Werte sind 0 oder 1

#### Funktion 2: split_dataset()
```python
def split_dataset(dataset, train_ratio=0.8):
    """
    Input:  dataset = komplettes Dataset
            train_ratio = 0.8 (80% Training, 20% Validation)
    Output: train_dataset, val_dataset
    
    Beispiel:
    - Berechne Anzahl Samples: total_size
    - train_size = int(total_size * train_ratio)
    - train_dataset = dataset.take(train_size)
    - val_dataset = dataset.skip(train_size)
    """
```

#### Funktion 3: prepare_dataset()
```python
def prepare_dataset(dataset, batch_size=8, shuffle=True, augment=False):
    """
    Input:  dataset = rohe Daten
            batch_size = 8 (bei wenig GPU Memory kleiner!)
            shuffle = True für Training, False für Validation
            augment = True für Training, False für Validation
    Output: Vorbereitetes dataset
    
    Pipeline für TRAINING:
    dataset = dataset.cache()                               # Im RAM speichern
    dataset = dataset.shuffle(buffer_size=1000)            # Mischen
    dataset = dataset.batch(batch_size)                     # In Batches gruppieren
    dataset = dataset.repeat()                              # Endlos wiederholen
    if augment:
        dataset = dataset.map(Augment())                    # Data Augmentation
    dataset = dataset.prefetch(tf.data.AUTOTUNE)           # Nächsten Batch vorbereiten
    
    Pipeline für VALIDATION (OHNE shuffle und augment):
    dataset = dataset.batch(batch_size)
    dataset = dataset.cache()
    dataset = dataset.prefetch(tf.data.AUTOTUNE)
    """
```

### Beispiel-Verwendung:
```python
# Daten laden
dataset = load_npy_dataset("path/to/img_tiles", "path/to/mask_tiles")

# Aufteilen
train_dataset, val_dataset = split_dataset(dataset, train_ratio=0.8)

# Vorbereiten
train_batches = prepare_dataset(train_dataset, batch_size=8, shuffle=True, augment=True)
val_batches = prepare_dataset(val_dataset, batch_size=8, shuffle=False, augment=False)
```

---

## 🧠 SCHRITT 3: Model Definition (U_net.py)

### Status: ⚠️ HAT FEHLER, MUSS KORRIGIERT WERDEN

### Hauptproblem:
Die Funktion `create_mobilenet_encoder()` gibt nichts zurück!

### Was fehlt:
```python
def create_mobilenet_encoder(input_shape=(512, 512, 4), trainable=False):
    # ... ganzer Code ...
    skip_outputs.append(x)  # Bottleneck hinzufügen
    
    # HIER FEHLT:
    return tf.keras.Model(inputs=inp, outputs=skip_outputs)
```

### Model Architektur:

```
Input (512, 512, 4)
        |
    [Encoder] - MobileNetV2 (vortrainiert)
        |
    [Bottleneck] (16, 16, filters)
        |
    [Decoder] - Upsample + Skip Connections
        |
Output (512, 512, 1) - Sigmoid Activation
```

### Skip Connections:
Die Skip Connections verbinden Encoder und Decoder:
```
Encoder                    Decoder
256x256 -----------------> Concat -> 256x256
128x128 -----------------> Concat -> 128x128
64x64   -----------------> Concat -> 64x64
32x32   -----------------> Concat -> 32x32
16x16 (Bottleneck) -----> Upsample
```

### Augment Klasse:
- Ist bereits implementiert ✅
- Macht horizontales Flipping
- Wichtig: Bild UND Maske werden ZUSAMMEN geflippt!

### Code-Organisation Problem:
Der `if __name__ == "__main__":` Block in U_net.py sollte NICHT da sein!
- U_net.py sollte nur Model-Definitionen enthalten
- Training gehört in train.py

---

## 🎯 SCHRITT 4: Training (train.py)

### Status: ⚠️ HAT FEHLER

### Aktuelle Probleme:

#### Problem 1: Daten laden
```python
# AKTUELL (funktioniert nicht):
images = tf.data.Dataset.load(dataset_path + "/img_tiles")

# SOLLTE SEIN:
from dataloader import load_npy_dataset, split_dataset, prepare_dataset
dataset = load_npy_dataset(img_path, mask_path)
```

#### Problem 2: Import fehlt
```python
# Am Anfang von train.py:
from U_net import build_unet, Augment
```

#### Problem 3: test_images nicht definiert
```python
# AKTUELL:
test_batches = test_images.batch(BATCH_SIZE)  # FEHLER!

# SOLLTE SEIN:
val_batches = prepare_dataset(val_dataset, batch_size=BATCH_SIZE, shuffle=False, augment=False)
```

#### Problem 4: Kein Validation Split
```python
# FEHLT:
train_dataset, val_dataset = split_dataset(dataset, train_ratio=0.8)
```

### Komplette Training Pipeline:

```python
# 1. DATEN LADEN
dataset = load_npy_dataset(img_path, mask_path)
train_dataset, val_dataset = split_dataset(dataset, train_ratio=0.8)
train_batches = prepare_dataset(train_dataset, batch_size=8, shuffle=True, augment=True)
val_batches = prepare_dataset(val_dataset, batch_size=8, shuffle=False, augment=False)

# 2. MODEL ERSTELLEN
model = build_unet(input_shape=(512, 512, 4), num_classes=1)
print(model.summary())

# 3. MODEL KOMPILIEREN
model.compile(
    optimizer=tf.keras.optimizers.Adam(learning_rate=1e-4),
    loss='binary_crossentropy',  # Für 2 Klassen (Vegetation ja/nein)
    metrics=['accuracy', tf.keras.metrics.IoU(num_classes=2, target_class_ids=[1])]
)

# 4. CALLBACKS DEFINIEREN
callbacks = [
    tf.keras.callbacks.ModelCheckpoint(
        'best_model.h5', 
        save_best_only=True, 
        monitor='val_loss'
    ),
    tf.keras.callbacks.EarlyStopping(
        patience=5, 
        monitor='val_loss'
    ),
    tf.keras.callbacks.ReduceLROnPlateau(
        factor=0.5, 
        patience=3
    )
]

# 5. TRAINING
history = model.fit(
    train_batches,
    epochs=20,
    steps_per_epoch=100,  # Anzahl Batches pro Epoch
    validation_data=val_batches,
    validation_steps=20,  # Anzahl Validation Batches
    callbacks=callbacks
)

# 6. MODEL SPEICHERN
model.save('final_model.h5')
```

---

## 📊 SCHRITT 5: Evaluierung & Visualisierung

### Was du nach dem Training machen solltest:

#### 1. Training History plotten
```python
plt.figure(figsize=(12, 4))

# Loss plotten
plt.subplot(1, 2, 1)
plt.plot(history.history['loss'], label='Training Loss')
plt.plot(history.history['val_loss'], label='Validation Loss')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.legend()

# Accuracy plotten
plt.subplot(1, 2, 2)
plt.plot(history.history['accuracy'], label='Training Accuracy')
plt.plot(history.history['val_accuracy'], label='Validation Accuracy')
plt.xlabel('Epoch')
plt.ylabel('Accuracy')
plt.legend()

plt.savefig('training_history.png')
```

#### 2. Vorhersagen visualisieren
```python
# Nimm ein Sample aus Validation Set
for images, masks in val_batches.take(1):
    predictions = model.predict(images)
    
    # Zeige erste 3 Samples
    for i in range(3):
        plt.figure(figsize=(15, 5))
        
        # Original Bild (nur RGB Kanäle)
        plt.subplot(1, 3, 1)
        plt.imshow(images[i, :, :, 1:4])  # RGB Kanäle
        plt.title('Original Bild')
        
        # Ground Truth Maske
        plt.subplot(1, 3, 2)
        plt.imshow(masks[i, :, :, 0], cmap='gray')
        plt.title('Ground Truth')
        
        # Vorhersage
        plt.subplot(1, 3, 3)
        plt.imshow(predictions[i, :, :, 0], cmap='gray')
        plt.title('Vorhersage')
        
        plt.savefig(f'prediction_{i}.png')
```

#### 3. Metriken berechnen
```python
from sklearn.metrics import classification_report, confusion_matrix

# Alle Vorhersagen sammeln
all_predictions = []
all_ground_truth = []

for images, masks in val_batches:
    preds = model.predict(images)
    all_predictions.append(preds > 0.5)  # Threshold bei 0.5
    all_ground_truth.append(masks)

# Flatten
predictions = np.concatenate(all_predictions).flatten()
ground_truth = np.concatenate(all_ground_truth).flatten()

# Metrics
print(classification_report(ground_truth, predictions))
print(confusion_matrix(ground_truth, predictions))
```

---

## 📝 Checkliste: Was du tun musst

### Sofort:
- [ ] **dataloader.py**: Schreibe die 3 fehlenden Funktionen
  - `load_npy_dataset()`
  - `split_dataset()`
  - `prepare_dataset()`

- [ ] **U_net.py**: Behebe den Fehler
  - Füge `return tf.keras.Model(inputs=inp, outputs=skip_outputs)` hinzu
  - Lösche den `if __name__ == "__main__":` Block (gehört nicht hier!)

- [ ] **train.py**: Korrigiere alle Fehler
  - Import: `from dataloader import load_npy_dataset, split_dataset, prepare_dataset`
  - Import: `from U_net import build_unet, Augment`
  - Ersetze `tf.data.Dataset.load()` durch `load_npy_dataset()`
  - Erstelle Validation Set
  - Entferne `test_images` und verwende `val_batches`

### Dann:
- [ ] Führe `image_snipper.py` aus (falls noch nicht gemacht)
- [ ] Teste Dataloader:
  ```python
  dataset = load_npy_dataset(img_path, mask_path)
  for img, mask in dataset.take(1):
      print("Image shape:", img.shape)
      print("Mask shape:", mask.shape)
  ```
- [ ] Teste Model:
  ```python
  model = build_unet()
  print(model.summary())
  ```
- [ ] Starte Training:
  ```python
  python train.py
  ```

### Danach:
- [ ] Evaluiere Ergebnisse
- [ ] Visualisiere Vorhersagen
- [ ] Optimiere Hyperparameter falls nötig

---

## 🔍 Typische Fehler & Lösungen

### Fehler 1: "Out of Memory"
**Lösung:** Reduziere `BATCH_SIZE` von 8 auf 4 oder 2

### Fehler 2: "Shapes don't match"
**Überprüfe:**
- Alle Bilder haben Shape (512, 512, 4)?
- Alle Masken haben Shape (512, 512, 1)?
- Batch hat richtige Form?

### Fehler 3: Model lernt nicht (Loss bleibt gleich)
**Mögliche Gründe:**
- Learning Rate zu groß oder zu klein
- Daten nicht richtig normalisiert
- Klassen-Ungleichgewicht (zu viele 0, zu wenige 1)

**Lösungen:**
- Versuche Learning Rate: 1e-3, 1e-4, 1e-5
- Prüfe Daten: `print(img.min(), img.max(), img.mean())`
- Verwende Class Weights oder Focal Loss

### Fehler 4: Training sehr langsam
**Überprüfe:**
- GPU wird verwendet? `tf.config.list_physical_devices('GPU')`
- Daten sind gecacht? `.cache()` in Pipeline
- Prefetch aktiviert? `.prefetch(tf.data.AUTOTUNE)`

---

## 💡 Tipps

1. **Starte klein**: Erst mit wenigen Bildern testen!
2. **Debug Mode**: Setze `epochs=1, steps_per_epoch=10` für schnelle Tests
3. **Visualisiere oft**: Schaue dir Predictions nach jedem Epoch an
4. **Speichere alles**: Model, Weights, History, Plots
5. **Dokumentiere**: Schreibe auf was du änderst und warum

---

## 📚 Nützliche Befehle

```python
# Anzahl Samples
total_samples = len(list(dataset))

# Dataset Info
for x, y in dataset.take(1):
    print(f"Image: {x.shape}, {x.dtype}, min={x.numpy().min():.3f}, max={x.numpy().max():.3f}")
    print(f"Mask: {y.shape}, {y.dtype}, unique values={np.unique(y.numpy())}")

# Model Info
print(model.summary())
print(f"Total params: {model.count_params():,}")

# GPU Check
print("GPUs available:", tf.config.list_physical_devices('GPU'))
```

---

**Viel Erfolg! 🚀**

Falls du Fragen hast, schaue in die TODO-Kommentare in den Dateien.
Mit dem Inline Chat kannst du dann die einzelnen Funktionen implementieren!
