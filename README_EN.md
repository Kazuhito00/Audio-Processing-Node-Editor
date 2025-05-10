[[Japanese](https://github.com/Kazuhito00/Audio-Processing-Node-Editor)/English] 

# Audio-Processing-Node-Editor
This is an audio processing app based on the node editor. <br>
It is intended for use in testing and comparing processing. <br>

<img src="https://github.com/user-attachments/assets/d59d10a4-8aef-4af4-af34-ef9ae6f682c7" loading="lazy" width="100%">

# Note
The nodes were added in the order that the creator (Takahashi) needed them, so<br>
There may be a temporary lack of nodes for basic audio processing.<br>

# Requirements
```
dearpygui            2.0.0     or later
onnx                 1.17.0    or later
onnxruntime          1.17.0    or later
opencv-python        4.11.0.86 or later
librosa              0.11.0    or later
sounddevice          0.5.1     or later
soundfile            0.13.1    or later
google-cloud-speech  2.32.0    or later ※Speech Recognition(Google Speech-to-Text)ノードを実行する場合
```

# Installation
1. Clone repository<br>`git clone https://github.com/Kazuhito00/Audio-Processing-Node-Editor`
1. Install package <br>`pip install -r requirements.txt`  
1. Run "main.py" <br>`python main.py`

# Usage
Here's how to run the app.
```bash
python main.py
```
* --setting<br>
Specify the path to the configuration file that contains the node size, sampling frequency, and Google credential path settings.<br>
default：node_editor/setting/setting.json

### Create Node
Select the node you want to create from the menu and click<br>
<img src="https://github.com/user-attachments/assets/4d9b810e-7e00-4084-b164-04412b093a60" loading="lazy" width="50%">

### Connect Node
Drag the output terminal to connect to the input terminal<br>
Only the same type set for the terminal can be connected<br>
<img src="https://github.com/user-attachments/assets/47f26d31-1e78-4185-810b-7073ec53dd9d" loading="lazy" width="50%">

### Delete Node
With the node you want to delete selected, press the "Del" key<br>
<img src="https://github.com/user-attachments/assets/d6f0e993-46f1-42a2-81a8-378ae0efc507" loading="lazy" width="50%">

### Export
Press "Export" from the menu and save the node settings(json file)<br>
<img src="https://github.com/user-attachments/assets/f3981cae-f58f-441e-b438-dee3ec83ed6d" loading="lazy" width="50%">

### Import
Read the node settings(json file) output by Export<br>
<img src="https://github.com/user-attachments/assets/3803a4a2-f60b-48a8-8ea0-3851bdfa1de8" loading="lazy" width="50%">

# Node
<details>
<summary>System Node</summary>

<table>
    <tr>
        <td width="200">
            Audio Control
        </td>
        <td width="320">
            <img src="https://github.com/user-attachments/assets/610e47b5-3b18-4c77-a864-9efd04ef52ce" loading="lazy" width="300px">
        </td>
        <td width="760">
            Nodes that control the Audio File node, Mic node, Noise node, and Write Wav File node<br>
            Only one Audio Control node can be created in a system.
        </td>
    </tr>
</table>
</details>

<details>
<summary>Input Node</summary>

<table>
    <tr>
        <td width="200">
            Audio File
        </td>
        <td width="320">
            <img src="https://github.com/user-attachments/assets/9fd91d47-e73a-461a-b3fc-eb4adac86e53" loading="lazy" width="300px">
        </td>
        <td width="760">
            A node that reads audio files (wav, mp3, ogg, m4a) and outputs chunk data.<br>
            "Select Audio File" button opens file dialog.
        </td>
    </tr>
    <tr>
        <td width="200">
            Mic
        </td>
        <td width="320">
            <img src="https://github.com/user-attachments/assets/1e576e21-42cb-4788-a8aa-70fdf82452cd" loading="lazy" width="300px">
        </td>
        <td width="760">
            A node that reads microphone input and outputs chunk data.<br>
            Select your microphone from the drop-down list.
        </td>
    </tr>
    <tr>
        <td width="200">
            Noise
        </td>
        <td width="320">
            <img src="https://github.com/user-attachments/assets/fd4d4dcf-a0ee-4e61-b07a-c7cc1bb8ecee" loading="lazy" width="300px">
        </td>
        <td width="760">
            A node that generates noise and outputs chunk data.<br>
            Select the noise type from the drop-down list(white noise, simple pink noise, hiss noise, hum noise, pulse noise).
        </td>
    </tr>
    <tr>
        <td width="200">
            Int Value
        </td>
        <td width="320">
            <img src="https://user-images.githubusercontent.com/37477845/172031284-95255053-6eaf-4298-a392-062129e698f6.png" loading="lazy" width="300px">
        </td>
        <td width="760">
            A node that outputs an integer value.<br>
        </td>
    </tr>
    <tr>
        <td width="200">
            Float Value
        </td>
        <td width="320">
            <img src="https://user-images.githubusercontent.com/37477845/172031323-98ae0273-7083-48d0-9ef2-f02af7fde482.png" loading="lazy" width="300px">
        </td>
        <td width="760">
            A node that outputs a float value.<br>
        </td>
    </tr>
</table>
</details>

<details>
<summary>Output Node</summary>

<table>
    <tr>
        <td width="200">
            Speaker
        </td>
        <td width="320">
            <img src="https://github.com/user-attachments/assets/bd29d949-41a2-4d5a-b1d7-61667ffad61c" loading="lazy" width="300px">
        </td>
        <td width="760">
            A node that receives chunk data and outputs it to speakers.<br>
            Select your speaker from the drop-down list.
        </td>
    </tr>
    <tr>
        <td width="200">
            Write Wav File
        </td>
        <td width="320">
            <img src="https://github.com/user-attachments/assets/0acf9e7d-3932-4ff6-a85e-3dcbdb38f406" loading="lazy" width="300px">
        </td>
        <td width="760">
            A node that receives chunk data and saves it as a Wav file.<br>
            The output destination is set to "output_directory" in "node_editor/setting/setting.json".<br>
            *The default is "./_output"
        </td>
    </tr>
</table>
</details>

<details>
<summary>Vol/Amp Node</summary>

<table>
    <tr>
        <td width="200">
            Gain Control(Scale Amplitude)
        </td>
        <td width="320">
            <img src="https://github.com/user-attachments/assets/c46f5ff0-081a-4197-85e3-6b247d443573" loading="lazy" width="300px">
        </td>
        <td width="760">
            A node that multiplies incoming chunk data by a constant and outputs the result.
        </td>
    </tr>
    <tr>
        <td width="200">
            Dynamic Range Compression
        </td>
        <td width="320">
            <img src="https://github.com/user-attachments/assets/628484e8-3cb5-47dd-8e54-15428c8e23ee" loading="lazy" width="300px">
        </td>
        <td width="760">
            A node that receives chunk data and outputs chunk data with dynamic range compression.<br>
            Threshold: Threshold value<br>
            Ratio: The ratio of values ​​that exceed the threshold to be compressed<br>
            Attack(ms): How quickly the gain is reduced when the threshold is exceeded (in milliseconds)<br>
            Release(ms): How quickly the gain is returned to normal when the value falls below the threshold (in milliseconds)<br>
        </td>
    </tr>
    <tr>
        <td width="200">
            Hard Limit
        </td>
        <td width="320">
            <img src="https://github.com/user-attachments/assets/4a3c542c-fc7c-4a88-a96d-833ad02917ce" loading="lazy" width="300px">
        </td>
        <td width="760">
            A node that receives chunk data and outputs chunk data with amplitude restriction.
        </td>
    </tr>
    <tr>
        <td width="200">
            Soft Limit(tanh)
        </td>
        <td width="320">
            <img src="https://github.com/user-attachments/assets/ba18788f-4acc-4e9d-8d13-ea973a3e3825" loading="lazy" width="300px">
        </td>
        <td width="760">
            A node that receives chunk data and outputs chunk data that has been subjected to gentle amplitude limiting using tanh.
        </td>
    </tr>
    <tr>
        <td width="200">
            Expander
        </td>
        <td width="320">
            <img src="https://github.com/user-attachments/assets/3252d730-84de-4124-b142-9671c5ee730c" loading="lazy" width="300px">
        </td>
        <td width="760">
            A node that receives chunk data and outputs chunk data with dynamic range expansion.<br>
            Threshold: Threshold value<br>
            Ratio: The ratio at which the value below the threshold is attenuated<br>
            Attack(ms): How quickly the gain is reduced when the threshold is exceeded (in milliseconds)<br>
            Release(ms): How quickly the gain is restored when the value returns below the threshold (in milliseconds)<br>
            Hold(ms): The grace period (in milliseconds) during which the gain is not immediately attenuated even if the value falls below the threshold<br>
        </td>
    </tr>
    <tr>
        <td width="200">
            Noise Gate
        </td>
        <td width="320">
            <img src="https://github.com/user-attachments/assets/697e99fe-862e-470c-b212-0b1194d0bea3" loading="lazy" width="300px">
        </td>
        <td width="760">
            A node that receives chunk data and outputs chunk data that has been noise gated.<br>
            Threshold: Threshold value<br>
            Attack(ms): How quickly the gain is reduced when the threshold is exceeded (in milliseconds)<br>
            Release(ms): How quickly the gain is restored when it falls back below the threshold (in milliseconds)<br>
            Hold(ms): Grace period during which the gain is not immediately attenuated even if it falls below the threshold (in milliseconds)
        </td>
    </tr>
</table>
</details>

<details>
<summary>FreqDomain Node</summary>

<table>
    <tr>
        <td width="200">
            Bandpass Filter(Butterworth IIR)
        </td>
        <td width="320">
            <img src="https://github.com/user-attachments/assets/fbd62390-5496-4e71-9e86-cca3fc33634d" loading="lazy" width="300px">
        </td>
        <td width="760">
            A node that receives chunk data and outputs chunk data passed through a bandpass filter (Butterworth IIR type).<br>
            High Cut Freq(Hz): Upper cutoff frequency (Hz)<br>
            Low Cut Freq(Hz): Lower cutoff frequency (Hz)<br>
            Filter Order: Filter order
        </td>
    </tr>
    <tr>
        <td width="200">
            Bandstop Filter(Butterworth IIR)
        </td>
        <td width="320">
            <img src="https://github.com/user-attachments/assets/a893e278-0b07-4e56-9877-d0413e4d5ad7" loading="lazy" width="300px">
        </td>
        <td width="760">
            A node that receives chunk data and outputs chunk data passed through a bandstop filter (Butterworth IIR type).<br>
            High Cut Freq(Hz): Upper cutoff frequency (Hz)<br>
            Low Cut Freq(Hz): Lower cutoff frequency (Hz)<br>
            Filter Order: Filter order
        </td>
    </tr>
    <tr>
        <td width="200">
            Highpass Filter(Butterworth IIR)
        </td>
        <td width="320">
            <img src="https://github.com/user-attachments/assets/c03f026d-baa2-4b47-b700-7a15eb565ee7" loading="lazy" width="300px">
        </td>
        <td width="760">
            A node that receives chunk data and outputs chunk data passed through a highpass filter (Butterworth IIR type).<br>
            Low Cut Freq(Hz): Lower cutoff frequency (Hz)<br>
            Filter Order: Filter order
        </td>
    </tr>
    <tr>
        <td width="200">
            Lowpass Filter(Butterworth IIR)
        </td>
        <td width="320">
            <img src="https://github.com/user-attachments/assets/eae66eb8-2c1c-4050-8855-e094b31ba964" loading="lazy" width="300px">
        </td>
        <td width="760">
            A node that receives chunk data and outputs chunk data passed through a lowpass filter (Butterworth IIR type).<br>
            High Cut Freq(Hz): Upper cutoff frequency (Hz)<br>
            Filter Order: Filter order
        </td>
    </tr>
    <tr>
        <td width="200">
            Simple EQ(Butterworth IIR)
        </td>
        <td width="320">
            <img src="https://github.com/user-attachments/assets/425bce4a-af9d-45d8-a3bb-f9953181e5c0" loading="lazy" width="300px">
        </td>
        <td width="760">
            A node that receives chunk data and outputs chunk data with gain amplification/attenuation of the target area.<br>
            High Cut Freq(Hz): Upper cutoff frequency (Hz)<br>
            Low Cut Freq(Hz): Lower cutoff frequency (Hz)<br>
            Gain(dB)：Gain(dB)
        </td>
    </tr>
    <tr>
        <td width="200">
            Simple Spectrogram
        </td>
        <td width="320">
            <img src="https://github.com/user-attachments/assets/2d9b8e18-1348-40ee-93d6-26f109fdeb82" loading="lazy" width="300px">
        </td>
        <td width="760">
            A node that receives chunk data and displays a simple spectrogram.<br>
            The shift size, window function (Hamming window or Hanning window), number of smoothings, etc. are set in "node_editor/setting/setting.json".<br>
            The lower end of the displayed data is 0Hz, and the upper end is the Nyquist frequency.
        </td>
    </tr>
</table>
</details>

<details>
<summary>TimeDomain Node</summary>

<table>
    <tr>
        <td width="200">
            Start Delay
        </td>
        <td width="320">
            <img src="https://github.com/user-attachments/assets/0a7a661e-0f9b-441c-af6c-70068162e6dc" loading="lazy" width="300px">
        </td>
        <td width="760">
            A node that receives chunk data and outputs the chunk data with a specified delay.<br>
            *Time specification must be done while the Audio Control node is in "Stop" mode.
        </td>
    </tr>
    <tr>
        <td width="200">
            Simple Mixer
        </td>
        <td width="320">
            <img src="https://github.com/user-attachments/assets/c8ed526a-404b-4ccb-aa1a-5a307a7dd00b" loading="lazy" width="300px">
        </td>
        <td width="760">
            A node that receives two chunk data and outputs the mixed chunk data.
        </td>
    </tr>
    <tr>
        <td width="200">
            Voice Activity Detection(Silero VAD)
        </td>
        <td width="320">
            <img src="https://github.com/user-attachments/assets/7df59ac0-2ee6-454b-93b5-31cb6127b262" loading="lazy" width="300px">
        </td>
        <td width="760">
            A node that receives chunk data and performs voice activity detection using Silero VAD.
        </td>
    </tr>
</table>
</details>

<details>
<summary>AudioEnhance Node</summary>

<table>
    <tr>
        <td width="200">
            Speech Enhancement(GTCRN)
        </td>
        <td width="320">
            <img src="https://github.com/user-attachments/assets/cecd2dc2-a673-4547-8549-4dabead84ee7" loading="lazy" width="300px">
        </td>
        <td width="760">
            A node that receives chunk data and outputs chunk data that has been enhanced using GTCRN.
        </td>
    </tr>
</table>
</details>

<details>
<summary>Analysis Node</summary>

<table>
    <tr>
        <td width="200">
            Speech Recognition(Google Speech-to-Text)
        </td>
        <td width="320">
            <img src="https://github.com/user-attachments/assets/6140903a-a048-47fe-8198-357f2807a4b7" loading="lazy" width="300px">
        </td>
        <td width="760">
            A node that receives chunk data and transcribes it using Google Speech-to-Text.<br>
            When using this node, please set the service account key in "google_application_credentials_json" in "node_editor/setting/setting.json"<br>
        </td>
    </tr>
</table>
</details>

<details>
<summary>Other Node</summary>

<table>
    <tr>
        <td width="200">
            Switch
        </td>
        <td width="320">
            <img src="https://github.com/user-attachments/assets/fd2a8bb7-ed7f-40c0-a632-eb6b5d77f22d" loading="lazy" width="300px">
        </td>
        <td width="760">
            A node that receives two chunk data and outputs the specified chunk data.
        </td>
    </tr>
    <tr>
        <td width="200">
            FPS
        </td>
        <td width="320">
            <img src="https://github.com/user-attachments/assets/a9e28170-98eb-4b72-b0ce-da23e7d05b21" loading="lazy" width="300px">
        </td>
        <td width="760">
            Node for measuring processing time.
        </td>
    </tr>
</table>
</details>

# Author
Kazuhito Takahashi(https://twitter.com/KzhtTkhs)
 
# License 
Audio-Processing-Node-Editor is under [Apache-2.0 license](LICENSE).<br><br>
The source code of Audio-Processing-Node-Editor itself is [Apache-2.0 license](LICENSE), but <br>
The source code for each algorithm is subject to its own license. <br>
For details, please check the LICENSE file included in each directory.
