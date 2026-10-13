# PowerAC

PowerAC is a local Minecraft anti-cheat focused on detecting AIM modules and aim-assist style cheats.

The neural network runs **entirely inside the plugin** (pure Java, no external server, no license key).
It analyses rotation telemetry in 64-sample windows and flags suspicious aim behaviour.

## Model files

All model files live in the plugin data folder (`plugins/PowerAC/`):

- `rotation.json` - the neural network (feature scaler, convolution/MLP weights, aggregation thresholds).
  A pre-trained model is bundled with the plugin and copied here on first start.
- `rotation_replay.json` - labelled training windows collected with `/powerac train` (created on demand).

## Training

The model can be fine-tuned in place, without any external service:

```text
/powerac train <legit|cheater> [true|false]
/powerac train <player> <legit|cheater> [true|false]
/powerac train stop [player]
/powerac train clear
```

Collect a batch of legit data and a batch of cheater data, and the plugin retrains and hot-swaps
the model automatically. Training runs asynchronously and never blocks the server thread.

## Commands

```text
/powerac player <name>            - player info
/powerac alerts                   - toggle alerts
/powerac monitor [player]         - live AI probabilities in the action bar
/powerac status                   - local model status
/powerac train ...                - train the local model
/powerac reload                   - reload config
```

## Author

Developed by `katze225 (tg @core2k21)`.

## License

Please read `LICENSE.md` before using, modifying, or distributing this source code.
