import 'bootstrap-icons/font/bootstrap-icons.css';
import {useRef} from "react";

function isLegalFile(fileType) {
    let parts = fileType.split('/');
    return parts[parts.length - 2] === "image";
}


function ChatInput({setListMessages}) {
    const fileInput = useRef(null)
    const textInput = useRef(null)

    function phraseCarDetails(details, key) {
        let l = ["מספר רכב", "פרטי/מסחרי", "יצרן", "רמת גימור", "קבוצת זיהום", "שנת ייצור", "דגם מנוע", "תאריך ביצוע טסט", "תוקף רשיון רכב", "בעלות", "מספר שילדה", "צבע", "צמיג קידמי", "צמיג אחורי", "סוג דלק", "עליה לכביש", "דגם"]
        let values = Object.values(details);
        let lines = Object.keys(values).map((key) => {
            return (<div key={key}>
                {`${values[key]} : ${l[key]}`}
            </div>);
        });

        return <div className="container_details">
            <div> Car {key + 1} details:</div>
            <br/>
            {lines}
            <br/>
        </div>;

    }

    async function sendToServer(message) {
        let response = await fetch("http://localhost:58112/api/chat/message?message=" + message, {
            method: 'GET', headers: {
                'Content-Type': 'application/json'
            },
        });
        response = await response.json();
        let add = {sent: false, type: "text", content: response[0].response}
        let newItem = {sent: true, type: "text", content: message};
        setListMessages(newItem, add);
    }

    const sendText = async (event) => {
        event.preventDefault();
        if (textInput.current.value.length === 0) {
            return;
        }
        let newItem = {sent: true, type: "text", content: textInput.current.value};
        let empty = {sent: false, type: "text", content: " "};
        setListMessages(newItem, empty);
        sendToServer(textInput.current.value);
        textInput.current.value = "";
    }

    const sendfile = async (event) => {
        event.preventDefault();
        if (!isLegalFile(event.target.files[0].type)) {
            alert("not image")
            return;
        }

        let image = {sender: true, type: "image", content: URL.createObjectURL(event.target.files[0])};
        let empty = {sent: false, type: "text", content: "trying to detect the car"};
        setListMessages(image, empty);
        const formData = new FormData();
        formData.append("image", event.target.files[0]);
        await fetch('http://localhost:58112/api/Chat/image', {
            method: 'POST',
            body: formData,
        }).then((response) => response.json())
            .then(data => {
                if (data.length === 0) {
                    let add = {
                        sent: false,
                        type: "text",
                        content: "I'm sorry but I can't detect the car from this image, Please send a new image"
                    }
                    setListMessages(image, add);
                    fileInput.current.value = null
                    return
                }
                let addValue = data.map((item, key) => {
                    let result = JSON.parse(item.response);
                    result = result.result.records[0]
                    delete result['_id'];
                    delete result['tozeret_cd'];
                    delete result['degem_cd'];
                    delete result['degem_nm'];
                    delete result['ramat_eivzur_betihuty'];
                    delete result['tzeva_cd'];
                    delete result['horaat_rishum'];
                    delete result['rank'];
                    return phraseCarDetails(result, key)
                })
                let content = <div>
                    <div>Detected {data.length} possible cars:</div>
                    {addValue}
                </div>
                let add = {sent: false, type: "text", content: content}
                setListMessages(image, add);
                fileInput.current.value = null;
            });


    }

    return (<div className="send-sec">
        <input type="file" accept="image/*" ref={fileInput} onChange={sendfile} style={{display: 'none'}}/>
        <button className="btn btn-success file-bottom" id="transparent-btn"
                onClick={() => {
                    fileInput.current.click();
                }}
        >
            <i className="bi bi-paperclip"/>
        </button>
        <input type="text" className="send-input" placeholder="massage" ref={textInput}/>
        <button className="btn btn-success send-bottom" id="transparent-btn" type="button" onClick={sendText}>
            <i className="bi bi-send"/>
        </button>

    </div>);
}

export default ChatInput;
