Vue.use(AsyncComputed)

var app = new Vue({
    el: "#app",
    data: {
        canvas: null,
        ctx: null,
        image: null,

        positiveMap:null,
        negativeMap:null,
        heatMap:null,
        firstPoint:[],
        secondPoint:[],
        mouseStatus:0,

        file_to_upload : null,

        image_to_annotate: null,

        current_page:1,
        num_page:1,
        images_per_page:10,
        images_this_page: [],
    },
    methods: {
        update_image : function(){
            Vue.http.get(
                '/api/image_list/'+this.current_page+'/'+this.images_per_page).
            then(response => {
                console.log(response.data);
                result = response.data;
                this.images_this_page = result.result;
                this.num_page = result.num_page
            }, response => {
                console.log(response);
            })
        },
        handleNewImage: function (url) {
            console.log(url);
            // var files = e.target.files || e.dataTransfer.files;
            // url = URL.createObjectURL(files[0]);
            var img = new Image();
            img.onload = function() {
                width = 500
                app.ctx.height = Math.floor(app.canvas.width / this.width * this.height)
                // console.log(this.width);
                // console.log(this.height);
                app.render();
            };
            img.src = url;
            this.image = img;
            this.render()
        },
        render: function () {
            if (this.image != null) {
                this.canvas.height = Math.floor(this.canvas.width / this.image.width * this.image.height)
                this.ctx.drawImage(this.image, 0, 0, this.canvas.width, this.ctx.height);
                // console.log('drawing image');
            }
            if (this.firstPoint.length >= 2 && this.secondPoint.length >= 2) {
                this.ctx.strokeStyle = "green";
                this.ctx.strokeRect(
                    Math.min(this.firstPoint[0], this.secondPoint[0]),
                    Math.min(this.firstPoint[1], this.secondPoint[1]),
                    Math.abs(this.secondPoint[0]-this.firstPoint[0]),
                    Math.abs(this.secondPoint[1]-this.firstPoint[1]));
            }
        },
        cvsClick: function(fuck){
            console.log(fuck);

            x = fuck.offsetX;
            y = fuck.offsetY;
            button = fuck.button;
            if (button != 0) {
                return ;
            }
            fuck.preventDefault();
            console.log(x, y, button);
            if (button === 0){
                this.positiveClick.push({x:x, y:y});
            } else if (button === 2){
                this.negativeClick.push({x:x, y:y});
            } else {
                console.error('fuck this')
            }
        },
        cvsUp: function(fuck){
            console.log(fuck);
            if(fuck.button != 0){
                return;
            }
            fuck.preventDefault();
            x = fuck.offsetX;
            y = fuck.offsetY;
            this.mouseStatus = 0;
            this.secondPoint = [x, y];
            this.render();
        },
        cvsMove: function(fuck){
            fuck.preventDefault();
            x = fuck.offsetX;
            y = fuck.offsetY;
            button = fuck.button;
            if (this.mouseStatus == 1){
                // console.log(x, y, button);
                this.secondPoint = [x, y];
                this.render();
            }
        },
        cvsDown: function(fuck){
            if(fuck.button != 0){
                return;
            }
            fuck.preventDefault();
            x = fuck.offsetX;
            y = fuck.offsetY;
            button = fuck.button;
            this.mouseStatus = 1;
            console.log(x, y, button);
            this.firstPoint = [x, y];
        },
        submit: function () {
            console.log('submit!');
            var form_data = new FormData();
            for(var i = 0; i < this.file_to_upload.length; i += 1){
                form_data.append(i, this.file_to_upload[i])
            }
            Vue.http.post('/api/add_img', form_data).then(response => {
                console.log(response);
            }, response => {
                console.log(response);
            })
        },
        submit_anno: function() {
            console.log('submit annotation');
            fp = [
                this.firstPoint[0]/this.canvas.width,
                this.firstPoint[1]/this.canvas.height,
            ];
            sp = [
                this.secondPoint[0]/this.canvas.width,
                this.secondPoint[1]/this.canvas.height,
            ];
            post_data = {
                image_url: this.image.src,
                points: [fp, sp],
            }
            console.log(post_data)
            Vue.http.post('/api/add_annotation', post_data)
        }
        // push_data: function(url, pictureName){
        //     if (this.image === null){
        //         alert('please select an image');
        //     }
        //     var formData = new FormData();
        //     formData.append('posclick', JSON.stringify(this.positiveClick));
        //     formData.append('negclick', JSON.stringify(this.negativeClick));
        //     formData.append('image', this.image);
        //     Vue.http.post(url , formData).then(response => {
        //         console.log(response);
        //         app[pictureName] = 'data:image/jpeg;base64,' + response.body.data;
        //     }, response =>{
        //         console.log(response);
        //     });
        // },
        // api_test: function () {
        //     fuckme = JSON.stringify({a:100,b:1134});
        //     Vue.http.post('/test', fuckme).then(
        //         response=> {
        //             console.log(response)
        //             result = JSON.parse(response.body);
        //             console.log(result);
        //             console.log(typeof result);
        //         }, response => {
        //             console.error(response)
        //         }
        //     )
        // },
        // api_img_test: function () {
        //     fuckme = JSON.stringify({a:200,b:100});
        //     Vue.http.post('/test_img', fuckme).then(
        //         response=> {
        //             console.log(response);
        //             result = response.body;
        //             console.log(result);
        //             app.positiveMap = 'data:image/jpeg;base64,'+result.data;
        //         }, response => {
        //             console.error(response)
        //         }
        //     )
        // }
    },
    // asyncComputed: {
    // },
    mounted() {
        this.canvas = document.getElementById('canvas');
        this.canvas.onmousedown = this.cvsDown;
        this.canvas.onmouseup = this.cvsUp;
        this.canvas.onmousemove = this.cvsMove;
        this.canvas.oncontextmenu=this.n;
        this.ctx = this.canvas.getContext('2d');
        this.update_image();
        console.log('init!')
    }
});
