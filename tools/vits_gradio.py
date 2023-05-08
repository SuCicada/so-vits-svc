import logging
import os
import sys

import gradio as gr

sys.path.append(os.path.join(os.path.dirname(__file__), "../"))

from tools.infer_base import SvcInfer

logging.getLogger('markdown_it').setLevel(logging.INFO)

# sys.path.append(os.path.abspath(os.path.dirname(os.path.dirname(__file__))))
print(sys.path)


class VitsGradio:
    svc_infer: SvcInfer = None

    def __init__(self):
        with gr.Blocks() as self.Vits:
            gr.Markdown("# Lain TTS with so-vits-svc-4.0 ")
            gr.Markdown("## web source in: [SuCicada/so-vits-svc](https://github.com/SuCicada/so-vits-svc/blob/4.0/tools/vits_gradio.py) ")
            gr.Markdown("## Models in: [SuCicada/Lain-so-vits-svc-4.0](https://huggingface.co/SuCicada/Lain-so-vits-svc-4.0/tree/main) ")

            with gr.Tabs():
                with gr.TabItem("tts"):
                    with gr.Row() as self.VoiceConversion:
                        self.text = gr.Textbox(label="input text", lines=5,
                                               value="こんにちは、私は人工知能です。私は、あなたの声を聞くことができます。")
                    with gr.Row():
                        with gr.Column():
                            with gr.Row():
                                self.tts_engine = gr.inputs.Dropdown(["edge-tts", "gtts", ], default="edge-tts",
                                                                     label="tts engine")
                                self.language = gr.inputs.Dropdown(["ja", "en", "zh", ], default="ja",
                                                                   label="language")
                                self.speed = gr.inputs.Slider(minimum=0, maximum=2, step=0.1, default=1,
                                                              label="speed")

                            self.tts_submit = gr.Button("Transform")
                        with gr.Column():
                            self.origin_output = gr.Audio(label="origin voice")
                    self.tts_target_output: gr.Audio = gr.Audio(label="lain voice")

                with gr.TabItem("svc"):
                    with gr.TabItem("voice to voice"):
                        self.svc_input_audio = gr.Audio(label="choose audio file")
                        self.vc_submit = gr.Button("voice conversion", variant="primary")

                    self.svc_target_output: gr.Audio = gr.Audio(label="lain voice")

                self.tts_submit.click(
                    self.tts_submit_func,
                    # self.get_audio(self.so_vits_svc1.get_audio_with_origin),
                    inputs=[self.text, self.tts_engine, self.language, self.speed],
                    outputs=[
                        self.origin_output,
                        self.tts_target_output,
                    ])

                self.vc_submit.click(
                    self.vc_submit_func,
                    inputs=[self.svc_input_audio],
                    outputs=[
                        self.svc_target_output,
                    ])

    def test(self, text, tts_engine):
        print(text, tts_engine)
        return text

    def tts_submit_func(self, text, tts_engine, language, speed):
        print(text, tts_engine, language)
        if self.svc_infer is None:
            self.svc_infer = SvcInfer()
        res = self.svc_infer.get_audio(
            tts_engine=tts_engine,
            text=text,
            language=language,
            speed=speed,
        )
        return res

    def vc_submit_func(self, svc_input_audio):
        print(svc_input_audio)
        if self.svc_infer is None:
            self.svc_infer = SvcInfer()
        sampling_rate, audio = svc_input_audio
        res = self.svc_infer.transform_audio(sampling_rate, audio)
        return res


grVits = VitsGradio()
demo = grVits.Vits

if __name__ == '__main__':
    demo.launch()