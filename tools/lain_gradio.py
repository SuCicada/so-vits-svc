import argparse
import logging
import os
import sys
from types import SimpleNamespace

import gradio as gr
import uvicorn
from gradio import networking, utils

sys.path.append(os.path.join(os.path.dirname(__file__), "../"))

from tools.infer_base import SvcInfer
from tools import infer_base, audio_utils

logging.getLogger('markdown_it').setLevel(logging.INFO)
logging.getLogger('urllib3.connectionpool').setLevel(logging.INFO)

# sys.path.append(os.path.abspath(os.path.dirname(os.path.dirname(__file__))))
print(sys.path)


class VitsGradio:
    svc_infer: SvcInfer = None
    options = SimpleNamespace()

    def __init__(self):
        with gr.Blocks() as self.Vits:
            gr.Markdown("<h1 style='text-align: center;'> Lain TTS with so-vits-svc-4.0 </h1>")
            gr.Markdown("<h2 style='text-align: center;'> Serial Experiments Lain </h1>")
            gr.Markdown(
                "## web source in: [SuCicada/so-vits-svc](https://github.com/SuCicada/so-vits-svc/blob/4.0/tools/lain_gradio.py) ")
            gr.Markdown(
                "## Models in: [SuCicada/Lain-so-vits-svc-4.0](https://huggingface.co/SuCicada/Lain-so-vits-svc-4.0/tree/main) ")

            with gr.Tabs():
                with gr.TabItem("tts"):
                    with gr.Row() as self.VoiceConversion:
                        self.text = gr.Textbox(label="input text", lines=5,
                                               value="人はみな、繋がっている。\n"
                                                     "リアルワールドとコンピュータネットワーク・ワイヤード。\n"
                                                     "シリアル　エクスペリメンツ　レイン。")
                    with gr.Row():
                        with gr.Column():
                            with gr.Row():
                                self.tts_engine = gr.inputs.Dropdown(["edge-tts", "gtts", ], default="gtts",
                                                                     label="tts engine")
                                self.language = gr.inputs.Dropdown(["ja", "en", "zh", ], default="ja",
                                                                   label="language")
                                self.speed = gr.inputs.Slider(minimum=0, maximum=2, step=0.1, default=1.1,
                                                              label="speed")

                            self.tts_submit = gr.Button("Transform")
                        with gr.Column():
                            self.origin_output = gr.Audio(label="origin voice")
                    self.tts_target_output: gr.Audio = gr.Audio(label="lain voice")

                with gr.TabItem("svc"):
                    with gr.TabItem("voice to voice"):
                        self.svc_input_audio = gr.Audio(label="choose audio file")
                        self.svc_input_bgm = gr.Audio(label="choose bgm file")
                        self.vc_submit = gr.Button("voice conversion", variant="primary")

                    self.svc_target_output: gr.Audio = gr.Audio(label="lain voice only")
                    self.merge_target_output: gr.Audio = gr.Audio(label="lain voice with bgm")

            gr.Markdown("## If you don't know what the following means, please don't change it.")
            with gr.Row(variant="panel"):
                with gr.Column():
                    gr.Markdown(value="""
                        <font size=2> 推理设置 </font>
                        """)
                    auto_predict_f0 = gr.Checkbox(
                        label="自动f0预测，配合聚类模型f0预测效果更好,会导致变调功能失效（仅限转换语音，歌声勾选此项会究极跑调）",
                        value=infer_base.auto_predict_f0)
                    F0_mean_pooling = gr.Checkbox(
                        label="是否对F0使用均值滤波器(池化)，对部分哑音有改善。注意，启动该选项会导致推理速度下降，默认关闭",
                        value=infer_base.F0_mean_pooling)
                    tran = gr.Number(label="变调（整数，可以正负，半音数量，升高八度就是12）",
                                     value=infer_base.tran)
                    cluster_infer_ratio = gr.Number(
                        label="聚类模型混合比例，0-1之间，0即不启用聚类。使用聚类模型能提升音色相似度，但会导致咬字下降（如果使用建议0.5左右）",
                        value=infer_base.cluster_infer_ratio)
                    slice_db = gr.Number(label="切片阈值", value=infer_base.slice_db)
                    noice_scale = gr.Number(label="noise_scale 建议不要动，会影响音质，玄学参数",
                                            value=infer_base.noice_scale)
                with gr.Column():
                    pad_seconds = gr.Number(
                        label="推理音频pad秒数，由于未知原因开头结尾会有异响，pad一小段静音段后就不会出现",
                        value=infer_base.pad_seconds)
                    clip_seconds = gr.Number(label="音频自动切片，0为不切片，单位为秒(s)",
                                             value=infer_base.clip_seconds)
                    lg_num = gr.Number(
                        label="两端音频切片的交叉淡入长度，如果自动切片后出现人声不连贯可调整该数值，如果连贯建议采用默认值0，注意，该设置会影响推理速度，单位为秒/s",
                        value=infer_base.lg_num)
                    lgr_num = gr.Number(
                        label="自动音频切片后，需要舍弃每段切片的头尾。该参数设置交叉长度保留的比例，范围0-1,左开右闭",
                        value=infer_base.lgr_num)
                    enhancer_adaptive_key = gr.Number(label="使增强器适应更高的音域(单位为半音数)|默认为0",
                                                      value=infer_base.enhancer_adaptive_key)
                    cr_threshold = gr.Number(
                        label="F0过滤阈值，只有启动f0_mean_pooling时有效. 数值范围从0-1. 降低该值可减少跑调概率，但会增加哑音",
                        value=infer_base.cr_threshold)

            self.tts_submit.click(
                self.tts_submit_func,
                # self.get_audio(self.so_vits_svc1.get_audio_with_origin),
                inputs=[self.text, self.tts_engine, self.language, self.speed,
                        auto_predict_f0, F0_mean_pooling, tran, cluster_infer_ratio, slice_db, noice_scale,
                        pad_seconds, clip_seconds, lg_num, lgr_num, enhancer_adaptive_key, cr_threshold],
                outputs=[
                    self.origin_output,
                    self.tts_target_output,
                ])

            self.vc_submit.click(
                self.vc_submit_func,
                inputs=[self.svc_input_audio, self.svc_input_bgm,
                        auto_predict_f0, F0_mean_pooling, tran, cluster_infer_ratio, slice_db, noice_scale,
                        pad_seconds, clip_seconds, lg_num, lgr_num, enhancer_adaptive_key, cr_threshold],
                outputs=[
                    self.svc_target_output,
                    self.merge_target_output,
                ])

    def test(self, text, tts_engine):
        print(text, tts_engine)
        return text

    def tts_submit_func(self, text, tts_engine, language, speed,
                        auto_predict_f0, F0_mean_pooling, tran, cluster_infer_ratio, slice_db, noice_scale,
                        pad_seconds, clip_seconds, lg_num, lgr_num, enhancer_adaptive_key, cr_threshold):

        print(text, tts_engine, language)
        if self.svc_infer is None:
            self.svc_infer = new_svc_infer()
        res = self.svc_infer.get_audio(
            tts_engine=tts_engine,
            text=text,
            language=language,
            speed=speed,

            auto_predict_f0=auto_predict_f0,
            F0_mean_pooling=F0_mean_pooling,
            tran=tran,
            cluster_infer_ratio=cluster_infer_ratio,
            slice_db=slice_db,
            noice_scale=noice_scale,

            pad_seconds=pad_seconds,
            clip_seconds=clip_seconds,
            lg_num=lg_num,
            lgr_num=lgr_num,
            enhancer_adaptive_key=enhancer_adaptive_key,
            cr_threshold=cr_threshold)
        return res

    def vc_submit_func(self, svc_input_audio, svc_input_bgm,
                       auto_predict_f0, F0_mean_pooling, tran, cluster_infer_ratio, slice_db, noice_scale,
                       pad_seconds, clip_seconds, lg_num, lgr_num, enhancer_adaptive_key, cr_threshold):
        print(svc_input_audio)
        if self.svc_infer is None:
            self.svc_infer = new_svc_infer()
        sampling_rate, audio = svc_input_audio
        target_sampling_rate, target_audio = self.svc_infer.transform_audio(
            sampling_rate, audio,

            auto_predict_f0=auto_predict_f0,
            F0_mean_pooling=F0_mean_pooling,
            tran=tran,
            cluster_infer_ratio=cluster_infer_ratio,
            slice_db=slice_db,
            noice_scale=noice_scale,

            pad_seconds=pad_seconds,
            clip_seconds=clip_seconds,
            lg_num=lg_num,
            lgr_num=lgr_num,
            enhancer_adaptive_key=enhancer_adaptive_key,
            cr_threshold=cr_threshold)

        if svc_input_bgm:
            bgm_sampling_rate, bgm_audio = svc_input_bgm
            combined_sampling_rate, combined_audio = audio_utils.merge_audio((target_sampling_rate, target_audio),
                                                                             (bgm_sampling_rate, bgm_audio))
        else:
            combined_sampling_rate, combined_audio = (target_sampling_rate, target_audio)
        return (target_sampling_rate, target_audio), (combined_sampling_rate, combined_audio)


print(sys.argv)
parser = argparse.ArgumentParser()
parser.add_argument('--model_path', type=str, help='model_path')
parser.add_argument('--config_path', type=str, help='config_path')
parser.add_argument('--cluster_model_path', type=str, help='cluster_model_path')
parser.add_argument("--hubert_model_path", type=str, help='hubert_model_path')
parser.add_argument("--port", type=int, help='port')
parser.add_argument('--debug', action='store_true', help='debug')
args = parser.parse_args()


def new_svc_infer():
    return SvcInfer(model_path=args.model_path,
                    config_path=args.config_path,
                    cluster_model_path=args.cluster_model_path,
                    hubert_model_path=args.hubert_model_path
                    )


grVits = VitsGradio()
demo = grVits.Vits


def main():
    if args.debug:
        port = networking.get_first_available_port(
            networking.INITIAL_PORT_VALUE,
            networking.INITIAL_PORT_VALUE + networking.TRY_NUM_PORTS,
        )
        original_path = sys.argv[0]
        abs_original_path = utils.abspath(original_path)
        path = os.path.normpath(original_path)
        path = path.replace("/", ".")
        path = path.replace("\\", ".")
        filename = os.path.splitext(path)[0]


        # gradio_folder = Path(inspect.getfile(gradio)).parent
        abs_parent: str = str(abs_original_path.parent)

        print("filename", filename, abs_parent)
        uvicorn.run(f"{filename}:demo.app", reload=True, reload_dirs=[abs_parent], port=port, log_level="warning")
    else:
        demo.launch(server_port=args.port,prevent_thread_lock=True)
    print("=====================================")
    print(f"server start at: http://127.0.0.1:{demo.server_port}")
    print("please open the url in your browser")
    print("enjoy it!")
    print("=====================================")
    demo.block_thread()

if __name__ == '__main__':
    main()
