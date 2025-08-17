(function(){
  const username = document.getElementById('username');
  const field    = document.getElementById('field');
  const out      = document.getElementById('out');
  const btnEnroll= document.getElementById('btn-enroll');
  const btnLogin = document.getElementById('btn-login');

  const kbaBox = document.getElementById('kba');
  const q1 = document.getElementById('q1');
  const q2 = document.getElementById('q2');
  const q3 = document.getElementById('q3');
  const a1 = document.getElementById('a1');
  const a2 = document.getElementById('a2');
  const a3 = document.getElementById('a3');
  const btnKBA = document.getElementById('btn-kba');

  const events = [];
  const now = () => performance.now();

  function push(type, e){
    if (e && (e.isComposing || e.repeat)) return;
    const obj = { type, t: now() };
    if (e) obj.code = e.code;
    events.push(obj);
  }

  field.addEventListener('keydown', e=> push('down', e));
  field.addEventListener('keyup',   e=> push('up', e));
  field.addEventListener('paste',   e=> push('paste')); // mark paste

  function resetCapture(){ events.length = 0; field.value = ''; }

  async function send(path, payload){
    const res = await fetch(`http://127.0.0.1:8000${path}`,{
      method:'POST', headers:{'Content-Type':'application/json'},
      body: JSON.stringify(payload)
    });
    return res.json();
  }

  btnEnroll.onclick = async ()=>{
    const u = username.value.trim();
    if(!u){ out.textContent = 'Enter username'; return; }
    const body = { username:u, events:[...events] };
    const data = await send('/enroll/sample', body);
    out.textContent = JSON.stringify(data, null, 2);
    resetCapture();
  };

  let pendingKBA = null;
  btnLogin.onclick = async ()=>{
    const u = username.value.trim();
    if(!u){ out.textContent = 'Enter username'; return; }
    const body = { username:u, events:[...events] };
    const data = await send('/auth/attempt', body);
    out.textContent = JSON.stringify(data, null, 2);

    if(data.status === 'KBA_REQUIRED'){
      pendingKBA = data.token;
      kbaBox.classList.remove('hidden');
      q1.textContent = data.questions[0];
      q2.textContent = data.questions[1];
      q3.textContent = data.questions[2];
    } else {
      kbaBox.classList.add('hidden');
      pendingKBA = null;
    }
    resetCapture();
  };

  btnKBA.onclick = async ()=>{
    if(!pendingKBA) return;
    const payload = { token: pendingKBA, a1:a1.value, a2:a2.value, a3:a3.value };
    const data = await send('/auth/kba', payload);
    out.textContent = JSON.stringify(data, null, 2);
    if(data.status === 'ALLOW'){
      kbaBox.classList.add('hidden');
      pendingKBA = null;
      a1.value = a2.value = a3.value = '';
    }
  };
})();
