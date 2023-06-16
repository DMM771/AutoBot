import {Link} from "react-router-dom";
import InputSec from "./InputSec";
import {useRef, useState} from "react";

function checkValidation(username, password, verifyPassword) {
    if (username.length === 0 || password.length === 0 || verifyPassword !== password) {
        alert("Input is invalid!")
        return false;
    }
    return true;
}

function RegisterPage() {
    let userid = {id: ""};
    const username = useRef(null);
    const password = useRef(null);
    const verifyPassword = useRef(null);
    const nextPage = useRef(null);
    const [error, setError] = useState(false);

    const reg = async (event) => {
        event.preventDefault();
        if (checkValidation(username.current.value, password.current.value, verifyPassword.current.value)) {
            const params = {
                username: username.current.value,
                password: password.current.value,
            }
            const request = {
                method: 'POST',
                headers: {'Content-Type': 'application/json'},
                body: JSON.stringify(params)
            };
            let res = await fetch('http://localhost:58112/api/Users/register', request);
            if (res.ok) {
                nextPage.current.click();
            } else setError(true);
        }
    }
    return (
        <div id="app" className="shadow log-reg-background">
            <Link to="/chat" state={userid} ref={nextPage}/>
            <InputSec text="Username" type="text" id={username}/>
            <InputSec text="Password" type="password" id={password}/>
            <InputSec text="Verify password" type="password" id={verifyPassword}/>
            {error && <div className="text-danger">Something wrong</div>}
            <div className="mb-3">
                <button type="submit" id="button" className="btn btn-primary btn-sm shadow" onClick={reg}>Register
                </button>
                <div id="register">Already registered? <Link to="/">Click here</Link> to login</div>
            </div>
        </div>
    );
}

export default RegisterPage;