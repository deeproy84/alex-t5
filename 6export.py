#For 3B and 11B models the export will take approximately 30-45 minutes.

export_dir = os.path.join(MODEL_DIR, "export")

model.batch_size = 1 # make one prediction per call
saved_model_path = model.export(
    export_dir,
    checkpoint_step=-1,  # use most recent
    beam_size=1,  # no beam search
    temperature=1.0,  # sample according to predicted distribution
)
print("Model saved to:", saved_model_path)