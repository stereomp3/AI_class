const input01 = document.getElementById("input01");
async function test(){
    // alert("qpt")
    // await window.fetch("test")
    if(input01.value != ""){
        const para = document.createElement("p");
        para.innerHTML = "user: "+input01.value+`</br>`;
        document.body.appendChild(para);
        var tmp = await window.fetch('getgpt', {
            body: '{"content":"'+input01.value+'"}',
            method: 'POST',
            headers: {
            'Content-Type': 'application/json'
            }
        }).then((response) => {
            //成功結果處理
            console.log(response);
            return response.json();
        })
        .catch((error) => {
            //錯誤結果處理
            console.log(error)
        })
        const para2 = document.createElement("p");
        para2.innerHTML = `GPT: `+tmp.get_res;
        document.body.appendChild(para2);
        
    }
    
}

