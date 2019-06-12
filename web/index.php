<link rel="stylesheet" href="https://maxcdn.bootstrapcdn.com/font-awesome/4.5.0/css/font-awesome.min.css"></link>
<h5>資料來源: https://foodsafety.family.com.tw/Web_FFD/Page/FFD_1_1.aspx?Category_id=17</h5>
<form action="upload.php" method="post" enctype="multipart/form-data" style="text-align: center; width:100%">
    <label class="btn btn-info" style="display: inline-block"> 
      <input type="file" name="file" id="file" accept="image/*" style="display:none"/> 
      <i class="fa fa-photo"></i> 上傳圖片 
    </label> 
</form>
<style type="text/css">
    .btn {
        width: 50%;
        max-width: 500px;
        display: inline-block;
        padding: 6px 12px;
        margin-bottom: 0;
        font-size: 14px;
        font-weight: 400;
        line-height: 1.42857143;
        text-align: center;
        white-space: nowrap;
        vertical-align: middle;
        -ms-touch-action: manipulation;
        touch-action: manipulation;
        cursor: pointer;
        -webkit-user-select: none;
        -moz-user-select: none;
        -ms-user-select: none;
        user-select: none;
        background-image: none;
        border: 1px solid transparent;
        border-radius: 4px;
    }

    .btn-info {
        color: #fff;
        background-color: #5bc0de;
        border-color: #46b8da;
    }

    .fa {
        display: inline-block;
        font: normal normal normal 14px/1 FontAwesome;
        font-style: normal;
        font-variant-ligatures: normal;
        font-variant-caps: normal;
        font-variant-numeric: normal;
        font-variant-east-asian: normal;
        font-weight: normal;
        font-stretch: normal;
        font-size: inherit;
        line-height: 1;
        font-family: FontAwesome;
        font-size: inherit;
        text-rendering: auto;
        -webkit-font-smoothing: antialiased;
        -moz-osx-font-smoothing: grayscale;
    }

    .fa-photo:before {
        content: "\f03e";
    }

</style>
<!-- from: loading.io -->
<style type="text/css">
  //from: loading.io
        @keyframes lds-rolling {
            0% {
                -webkit-transform: translate(-50%, -50%) rotate(0deg);
                transform: translate(-50%, -50%) rotate(0deg);
            }
            100% {
                -webkit-transform: translate(-50%, -50%) rotate(360deg);
                transform: translate(-50%, -50%) rotate(360deg);
            }
        }

        @-webkit-keyframes lds-rolling {
            0% {
                -webkit-transform: translate(-50%, -50%) rotate(0deg);
                transform: translate(-50%, -50%) rotate(0deg);
            }
            100% {
                -webkit-transform: translate(-50%, -50%) rotate(360deg);
                transform: translate(-50%, -50%) rotate(360deg);
            }
        }

        .lds-rolling {
            position: relative;
        }

        .lds-rolling div,
        .lds-rolling div:after {
            position: absolute;
            width: 60px;
            height: 60px;
            border: 10px solid #1d3f72;
            border-top-color: transparent;
            border-radius: 50%;
        }

        .lds-rolling div {
            -webkit-animation: lds-rolling 1s linear infinite;
            animation: lds-rolling 1s linear infinite;
            top: 50px;
            left: 50px;
        }

        .lds-rolling div:after {
            -webkit-transform: rotate(90deg);
            transform: rotate(90deg);
        }

        .lds-rolling {
            width: 100px !important;
            height: 100px !important;
            -webkit-transform: translate(-100px, -100px) scale(1) translate(100px, 100px);
            transform: translate(-100px, -100px) scale(1) translate(100px, 100px);
        }
  .lds-css {
    text-align: center;
  }

    </style>

<div class="lds-css" id="loading" style="display:none;">
    <div style="display: inline-block" class="lds-rolling">
        <div></div>
    </div>
</div>
<!-- end from: loading.io -->
<div class="lds-css">
    <div style="display: inline-block">
        <a id="link"><img id="image"/></a>
    </div>
</div>

<script> 
dirnks = ["午後時光重乳奶茶",
    "可爾必思",
    "生活紅茶",
    "生活綠茶",
    "立頓奶茶",
    "貝納頌重乳拿鐵",
    "阿薩姆奶茶",
    "麥香奶茶(15元)",
    "麥香紅茶(15元)",
    "麥香綠茶(10元)",
    "木瓜牛奶",
    "西瓜牛奶",
    "咖啡廣場",
    "阿華田",
    "雀巢檸檬紅茶"
] 
  links = ["https://foodsafety.family.com.tw/Web_FFD/Page/FFD_1_2.aspx?CatNa=A4C0BB92860F4074AC8D5C3D1DD79ED1&CatNo=82DCCF8D870D9B18&CmnoCode=721E599A9364FD4F&Img=003004.3140327.jpg",
    "https://foodsafety.family.com.tw/Web_FFD/Page/FFD_1_2.aspx?CatNa=A4C0BB92860F4074AC8D5C3D1DD79ED1&CatNo=82DCCF8D870D9B18&CmnoCode=2F3275D2EA12D0BF&Img=003004.3330021.jpg",
    "https://foodsafety.family.com.tw/Web_FFD/Page/FFD_1_2.aspx?CatNa=A4C0BB92860F4074AC8D5C3D1DD79ED1&CatNo=82DCCF8D870D9B18&CmnoCode=3BF36FF6880895A6&Img=003004.3110071.jpg",
    "https://foodsafety.family.com.tw/Web_FFD/Page/FFD_1_2.aspx?CatNa=A4C0BB92860F4074AC8D5C3D1DD79ED1&CatNo=82DCCF8D870D9B18&CmnoCode=0B0034065AB4BF3F&Img=003004.3120050.jpg",
    "https://foodsafety.family.com.tw/Web_FFD/Page/FFD_1_2.aspx?CatNa=A4C0BB92860F4074AC8D5C3D1DD79ED1&CatNo=82DCCF8D870D9B18&CmnoCode=902AF342B8202DFF&Img=003004.3140240.jpg",
    "https://foodsafety.family.com.tw/Web_FFD/Page/FFD_1_2.aspx?CatNa=A4C0BB92860F4074AC8D5C3D1DD79ED1&CatNo=82DCCF8D870D9B18&CmnoCode=1A74B2D526F2CD3D&Img=003004.3310213.jpg",
    "https://foodsafety.family.com.tw/Web_FFD/Page/FFD_1_2.aspx?CatNa=A4C0BB92860F4074AC8D5C3D1DD79ED1&CatNo=82DCCF8D870D9B18&CmnoCode=C68983C8C00D335B&Img=003004.3140418.jpg",
    "https://foodsafety.family.com.tw/Web_FFD/Page/FFD_1_2.aspx?CatNa=A4C0BB92860F4074AC8D5C3D1DD79ED1&CatNo=82DCCF8D870D9B18&CmnoCode=C70ED69990E48D73&Img=003004.3140159.jpg",
    "https://foodsafety.family.com.tw/Web_FFD/Page/FFD_1_2.aspx?CatNa=A4C0BB92860F4074AC8D5C3D1DD79ED1&CatNo=82DCCF8D870D9B18&CmnoCode=20F54DD310B6A94D&Img=003004.3110067.jpg",
    "https://foodsafety.family.com.tw/Web_FFD/Page/FFD_1_2.aspx?CatNa=A4C0BB92860F4074AC8D5C3D1DD79ED1&CatNo=82DCCF8D870D9B18&CmnoCode=3C4802B074790C89&Img=003004.3122034.jpg",
    "https://foodsafety.family.com.tw/Web_FFD/Page/FFD_1_2.aspx?CatNa=5498A6DC29DA94FC&CatNo=A3DF7304306642EB&CmnoCode=778D1D1A00FD7F1E&Img=003004.2820025.jpg",
    "https://foodsafety.family.com.tw/Web_FFD/Page/FFD_1_2.aspx?CatNa=5498A6DC29DA94FC&CatNo=A3DF7304306642EB&CmnoCode=67C6DD5D81318C2C&Img=003004.2820689.jpg",
    "https://foodsafety.family.com.tw/Web_FFD/Page/FFD_1_2.aspx?CatNa=A4C0BB92860F4074AC8D5C3D1DD79ED1&CatNo=82DCCF8D870D9B18&CmnoCode=40D2FB76CC9D541B&Img=003004.3310481.jpg",
    "https://foodsafety.family.com.tw/Web_FFD/Page/FFD_1_2.aspx?CatNa=5498A6DC29DA94FC&CatNo=A3DF7304306642EB&CmnoCode=B99A2D854652C4A0&Img=003004.2820075.jpg",
    "https://foodsafety.family.com.tw/Web_FFD/Page/FFD_1_2.aspx?CatNa=A4C0BB92860F4074AC8D5C3D1DD79ED1&CatNo=82DCCF8D870D9B18&CmnoCode=56E78EB97DEEB765&Img=003004.3110449.jpg"
]
const fileBtn = document.getElementById('file') 
fileBtn.addEventListener('change', () => {
  if (!file.files[0]) 
    return 
  document.getElementById('loading').style = ""
  file.disabled = true
  const fd = new FormData() 
  fd.append('file', file.files[0]) 
  fetch('upload.php', {
            method: 'POST',
            body: fd
        }).then(res => {
            if (res.ok) {
                res.text().then(res => {
                  getResult(res)
                });
            } else {
                document.getElementById('loading').style = "display:none;"
                file.disabled = false 
                res.text().then(res => {
                  alert(res)
                });
            }
        })
  file.value = ""
    }

)
function getResult(res) {
    fetch('/result/' + res).then(result => {
        if (result.status != 200) 
          window.setTimeout(getResult, 500, res)
        else {
            document.getElementById('loading').style = "display:none;"
            file.disabled = false 
            result.text().then(R => {
                document.getElementById('image').src = R+".jpg"
                document.getElementById('link').href=links[R]
                //alert(dirnks[R])
            })
        }
    })
}

</script>