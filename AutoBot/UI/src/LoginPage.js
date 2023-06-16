import {Link} from "react-router-dom";
import InputSec from "./InputSec";
import {useRef, useState} from "react";


async function checkIfRegistered(username, password) {
    const params = {
        username: username,
        password: password
    }
    const request = {
        method: 'POST',
        headers: {
            'Content-Type': 'application/json'
        },
        body: JSON.stringify(params),
    };
    let res = await fetch("http://localhost:58112/api/Users/login", request);
    return res.ok;
}

async function login(username, password) {
    if (password.current.value.length < 8 || username.current.value.length === 0) {
        alert("Input is invalid!");
        return false;
    }

    return await checkIfRegistered(username.current.value, password.current.value);
}


function LoginPage() {
    let userid={id:""};
    const username = useRef(null);
    const password = useRef(null);
    const nextPage = useRef(null);
    let [error, setError] = useState(false);
    const check = async (event) => {
        event.preventDefault();
        const isLog = await login(username, password)
        if (isLog) {
            nextPage.current.click();
        } else {
            setError(true);
        }
    }
    return (
        <div id="app" className="shadow log-reg-background">
            <header>MESSENGER</header>
            <Link to="/chat" state={userid} ref={nextPage}/>
            <InputSec text="Username" type="text" id={username}/>
            <InputSec text="Password" type="password" id={password}/>
            {error && <div className="text-danger">username or password wrong</div>}
            <div className="mb-3 mt-2">
                <button type="button" className="btn btn-primary btn-sm shadow" onClick={check}>Login
                </button>
                <div id="register">Not registered? <Link to="/register">Click here</Link> to register</div>
            </div>
        </div>);
}

export default LoginPage;