import Message from "./message";
import {useEffect, useRef} from "react";


function ChatBox({listMessages}) {
    let window = useRef(null);

    useEffect(() => {
        const chatWindow = window.current;
        chatWindow.scrollTop = chatWindow.scrollHeight;
    }, [listMessages]);
    var dataList;
    if (listMessages) {
        dataList = listMessages.map((message, key) => {
            if (key === listMessages.length - 1 && message.content !== "" && listMessages.length !== 1) {
                return (
                    <Message
                        sender={message.sent}
                        type={message.type}
                        value={message.content}
                        isTypingAnimation={true}
                        key={key}
                    />
                );
            } else
                return (
                    <Message
                        sender={message.sent}
                        type={message.type}
                        value={message.content}
                        key={key}
                    />
                );
        });

        dataList.reverse();
    }

    return (
        <div className="chat-box scrollbar" ref={window}>
            {dataList}
        </div>
    )
}

export default ChatBox;