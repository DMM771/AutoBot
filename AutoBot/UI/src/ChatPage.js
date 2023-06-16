import ChatBox from "./chatBox";
import ChatInput from "./ChatInput";
import {useState} from "react";


function ChatPage() {
    const [listMessages, setListMessages] = useState([{sent: false, type: "text", content: "hi, how can i help you?"}]);
    const handleSend = (item1,item2) => {
        if (item2){
            const newList = [...listMessages, item1,item2]
            setListMessages(newList);
            return
        }
        const newList = [...listMessages, item1]
        setListMessages(newList);
    }
    return (
        <div className="chat-page-background">
            <div className="chat-sec">
                <nav className="navbar-light px-3 sticky-sm-top chat-nav fw-bolder big-text" id="chatNav">
                    AutoBot
                </nav>
                <ChatBox listMessages={listMessages}/>
                <ChatInput setListMessages={handleSend}/>
            </div>
        </div>
    )
}

export default ChatPage;