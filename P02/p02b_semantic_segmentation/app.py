"""PetCut local. Ejecutar después de entrenar el modelo con skips."""
import argparse
import gradio as gr
from src.inference import PetCut


def build_app(checkpoint, device="cpu"):
    petcut = PetCut(checkpoint, device)
    with gr.Blocks(title="PetCut") as demo:
        gr.Markdown("# PetCut\nRecorta tu mascota y transforma el fondo. Mejor resultado con un gato o perro visible.")
        with gr.Row():
            photo = gr.Image(type="pil", label="Fotografía")
            edited = gr.Image(type="pil", label="Resultado (descargable)", format="png")
        effect = gr.Dropdown(["Desenfocar fondo", "Fondo de color", "Fondo gris", "Sticker"],
                             value="Desenfocar fondo", label="Efecto")
        color = gr.ColorPicker(value="#86b6d8", label="Color del fondo")
        blur = gr.Slider(0, 30, value=12, step=1, label="Desenfoque")
        button = gr.Button("Crear recorte", variant="primary")
        with gr.Row():
            mask = gr.Image(type="pil", label="Máscara (blanco: mascota)", format="png")
            overlay = gr.Image(type="pil", label="Superposición")
        slider = gr.ImageSlider(label="Original / resultado", type="pil")
        # TODO T6_APP: Conecta button.click a PetCut con los cuatro inputs y outputs; maneja imagen None con gr.Error.
        raise NotImplementedError("T6_APP: Conecta button.click a PetCut con los cuatro inputs y outputs; maneja imagen None con gr.Error.")
    return demo


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", default="results/skip/best.pt")
    parser.add_argument("--device", choices=["cpu", "cuda"], default="cpu")
    parser.add_argument("--port", type=int, default=7860)
    args = parser.parse_args()
    build_app(args.checkpoint, args.device).launch(server_name="127.0.0.1", server_port=args.port, share=False)
