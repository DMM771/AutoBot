function InputSec({text, type, id}) {
    return (
        <div className="big-text">
            <span>{text}</span>
            <input type={type} className="form-control" ref={id}/>
        </div>
    )
}

export default InputSec;