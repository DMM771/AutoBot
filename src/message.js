function Message({sender, type, value,isTypingAnimation}) {
    let offset = ""
    if (type === "text") {
        if (!sender) {
            offset = "offset-6"
        }
        return (
            <div className="row message">
                <div className={`col-6 ${offset}`}>
                    <div className={`text-break ${isTypingAnimation ? 'typing-animation' : ''} `} id="message-border">
                        {value}
                    </div>
                </div>
            </div>
        )
    }
    if (type === "image") {
        if (sender) {
            offset = "offset-10"
        }
        return (
            <div className="row message">
                <div className={`col-2 ${offset}`}>
                    <div className="text-break" id="message-border">
                        <img src={value} className="message-image" alt="not found"/>
                    </div>
                </div>
            </div>
        )
    }
}
export default Message;