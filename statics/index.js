Vue.use(AsyncComputed)


class Box {
    constructor(){
        this.x=0;
        this.width=0;

        this.y=0;
        this.height=0;
        this.class = -1;

        
        this.dx=0;
        this.dy=0;
        
        this.status=0;
        // 0: not selected
        // 10: moving status
        // 20: resizing status

        this.boder_range = 5
        this.event = null;
        this.mouse_down_event = null;
    }

    get top_left() {
        if (this.status == 0) {
            return [this.x - this.width / 2,
                    this.y - this.height / 2,
                    this.width,
                    this.height]
        } else if (this.status == 10) {
            var dx = this.mouse_down_event.offsetX - this.event.offsetX;
            var dy = this.mouse_down_event.offsetY - this.event.offsetY;
            return [this.x - this.width / 2 - dx,
                this.y - this.height / 2 - dy,
                this.width,
                this.height]
        } else if (this.status == 20) {
            var dx = this.event.offsetX - this.mouse_down_event.offsetX;
            var dy = this.event.offsetY - this.mouse_down_event.offsetY;
            var p1 = [this.x - this.width / 2, this.y - this.height / 2];
            var p2 = [this.x + this.width / 2, this.y + this.height / 2];
            p2[0] += dx;
            p2[1] += dy;
            return [Math.min(p1[0], p2[0]),
                    Math.min(p1[1], p2[1]),
                    Math.abs(p1[0] - p2[0]),
                    Math.abs(p1[1] - p2[1])]
        }
    }

    get bbox() {
        return [this.x, this.y, this,this.width, this.height];
    }

    get _in() {
        var x = event.offsetX;
        var y = event.offsetY;
        return (this.width / 2 - Math.abs(this.x - x) > 0) &&
                (this.height / 2 - Math.abs(this.y - y) > 0);
    }

    get _boder() {
        var x = event.offsetX;
        var y = event.offsetY;
        return this._in && ((Math.abs(this.width / 2 - Math.abs(this.x - x)) < this.boder_range) ||
               (Math.abs(this.height / 2 - Math.abs(this.y - y)) < this.boder_range));
    }

    _update_position(){
        if(this.status == 20){
            var dx = this.event.offsetX - this.mouse_down_event.offsetX;
            var dy = this.event.offsetY - this.mouse_down_event.offsetY;
            var p1 = [this.x - this.width / 2, this.y - this.height / 2];
            var p2 = [this.x + this.width / 2, this.y + this.height / 2];
            p2[0] += dx;
            p2[1] += dy;
            this.width = Math.abs(p1[0] - p2[0]);
            this.height = Math.abs(p1[1] - p2[1]);
            this.x = (p1[0] + p2[0]) / 2;
            this.y = (p1[1] + p2[1]) / 2;
        } else if (this.status == 10) {
            var dx = this.event.offsetX - this.mouse_down_event.offsetX;
            var dy = this.event.offsetY - this.mouse_down_event.offsetY;
            this.x += dx;
            this.y += dy;
        }
    }

    handleEvent(event) {
        this.event = event;
        var block_event = false;
        switch(this.status){
        case 0:
            if (event.type == 'mousemove'){}
            if (event.type == 'mouseup'){}
            if (event.type == 'mousedown'){
                if (this._boder) {
                    this.status = 20;
                    this.mouse_down_event = event;
                    block_event = true;
                } else if (this._in) {
                    this.status = 10;
                    this.mouse_down_event = event;
                    block_event = true;
                }
            }
            break;
        case 10:
            block_event = true;
            if (event.type == 'mouseup') {
                this._update_position();
                block_event = false;
                this.status = 0;
            }
            break;
        case 20:
            block_event = true;
            if (event.type == 'mouseup') {
                this._update_position();
                block_event = false;
                this.status = 0;
                console.log('cancel event');
            }
        };
        return block_event;
    }

}

var app = new Vue({
    el: "#app",
    data: {
        canvas: null,
        ctx: null,
        image: null,

        current_box:-1,
        boxes:[],
        adding_box: false,

        file_to_upload : null,

        current_image: null,

        current_page:1,
        images_per_page:10,
        num_page:0,
        images_this_page: [],

        sessions:[],
        current_session:{"session_name":'null'},

        new_session_name: null,
    },
    methods: {
        update_session : function(){
            Vue.http.get(
                '/api/sessions').
            then(response => {
                console.log(response.data);
                result = response.data;
                this.sessions = response.data.sessions;
            }, response => {
                console.log(response);
            })
        },
        add_session: function(){
            console.log('adding fuck session');
            Vue.http.post('/api/session', {
                session_name:this.new_session_name,
            }).then(response => {
                console.log(response.data);
                this.update_session();
            })
        },
        handleNewImage: function (record) {
            this.current_image = record;
            // set image
            url = record.url;
            var img = new Image();
            img.onload = function() {
                const width = 500;
                console.log(this);
                console.log(this.width);
                app.canvas.height = Math.floor(app.canvas.width / this.width * this.height)
                app.render();
            };
            img.src = url;
            this.image = img;
            // add boxes
            Vue.http.get()
        },
        render: function () {
            if (this.image != null) {
                this.canvas.height = Math.floor(this.canvas.width / this.image.width * this.image.height)
                this.ctx.drawImage(this.image, 0, 0, this.canvas.width, this.canvas.height);
            }
            for(var i=0; i < this.boxes.length; i += 1){
                this.ctx.strokeStyle = "green";
                if (this.current_box <0) {
                    if (this.boxes[i]._boder) {
                        this.ctx.strokeStyle = 'red';
                    } else if (this.boxes[i]._in){
                        this.ctx.fillStyle = 'rgba(0, 255, 0, 0.2)';
                        this.ctx.fillRect(
                            ...this.boxes[i].top_left
                        );        
                    }
                }
                this.ctx.strokeRect(
                    ...this.boxes[i].top_left
                );
            }
        },
        handle_mouse_event(event){
            if (this.adding_box){
            if(event.type == 'mousedown'){
                console.log(event);
                this.adding_box = false;
                newbox = new Box();
                newbox.x = event.offsetX;
                newbox.y = event.offsetY;
                newbox.mouse_down_event = event;
                newbox.status = 20;
                newbox.dx = 1;
                newbox.dy = 1;
                this.boxes.push(newbox)
            }}
            if (this.current_box < 0) {
                for(var i = 0; i < this.boxes.length; i += 1){
                    if (this.boxes[i].handleEvent(event)){
                        this.current_box = i;
                        console.log('set box to ' + this.current_box);
                        break;
                    }
                }
            } else {
                if (!this.boxes[this.current_box].handleEvent(event)) {
                    this.current_box = -1;
                    console.log('set box to ' + this.current_box);
                }
            }
            event.preventDefault();
            this.render();
        },
        _normal_post_data: function name(url, form_data) {
            form_data.set('session_name', this.current_session.session_name)
            Vue.http.post(url, form_data).then(response => {
                console.log(response);
            }, response => {
                console.log(response);
            });
        },
        submit: function () {
            console.log('submit!');
            var form_data = new FormData();
            for(var i = 0; i < this.file_to_upload.length; i += 1){
                form_data.append(i, this.file_to_upload[i])
                if (i != 0 && i % 10 == 0) {
                    this._normal_post_data('/api/add_img', form_data);
                    form_data = new FormData();
                }
            }
            if (i % 10 !=0){
                this._normal_post_data('/api/add_img', form_data);
            }
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
            point = [
                (this.firstPoint[1] + this.secondPoint[1]) / 2 / this.canvas.height,
                (this.firstPoint[0] + this.secondPoint[0]) / 2 / this.canvas.width,
                Math.abs(this.firstPoint[1] - this.secondPoint[1]) / 2 / this.canvas.height,
                Math.abs(this.firstPoint[0] - this.secondPoint[0]) / 2 / this.canvas.width,
            ]
            post_data = {
                image_url: this.image.src,
                points: point,
            }
            console.log(post_data)
            Vue.http.post('/api/add_annotation', post_data)
        }
    },
    asyncComputed: {
        images_this_page() {
            Vue.http.get('/api/image_list/'+this.current_session.session_name+'/'+this.current_page+'/'+this.images_per_page)
            .then(response => {
                console.log(response.data);
                this.num_page = response.data.num_page;
                this.images_this_page = response.data.result;
                return response.data.result;
            })
        },
    },
    computed: {

    },
    watch: {
        // image_to_annotate: function(val) {
            
        // }
    },
    mounted() {
        this.canvas = document.getElementById('canvas');
        this.canvas.onmousedown = this.handle_mouse_event;
        this.canvas.onmouseup = this.handle_mouse_event;
        this.canvas.onmousemove = this.handle_mouse_event;
        this.canvas.oncontextmenu=this.n;
        this.ctx = this.canvas.getContext('2d');
        this.update_session();
        console.log('init!')
    }
});
