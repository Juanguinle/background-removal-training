import torch
import segmentation_models_pytorch as smp
import coremltools as ct

if __name__ == '__main__':
    # Cargar modelo entrenado
    model = smp.DeepLabV3Plus(
        encoder_name='mobilenet_v2',
        encoder_weights=None,
        in_channels=3,
        classes=1,
        activation=None
    )
    model.load_state_dict(torch.load('C:/train_bg_removal/models/best_model.pth'))
    model.eval()

    # Ejemplo input para tracing
    example_input = torch.rand(1, 3, 512, 512)

    # Trace modelo
    print("Tracing modelo...")
    traced_model = torch.jit.trace(model, example_input)

    # Convertir a Core ML
    print("Convirtiendo a Core ML...")
    mlmodel = ct.convert(
        traced_model,
        inputs=[ct.ImageType(name="input", shape=(1, 3, 512, 512))],
        convert_to="mlprogram",
        compute_precision=ct.precision.FLOAT16
    )

    # Guardar
    mlmodel.save('C:/train_bg_removal/models/PersonSegmentation.mlpackage')
    print('Modelo Core ML guardado en C:/train_bg_removal/models/PersonSegmentation.mlpackage')
