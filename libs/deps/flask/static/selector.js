$(window).on("load", function () {
    $.get("/cameras", function (data) {
        $("#cam-select").html(
            `<option value="-1" selected disabled hidden>No camera selected...</option>` +
                data
        );
        adjust_option_width();
    });

    $("#cam-select").on("change", function () {
        $.ajax({
            url: "/camera/" + this.value,
            type: "POST",
            dataType: "text",
            success: function (data) {
                console.log("success " + data);
                $("#reso-select").html(
                    `<option value="-1" selected disabled hidden>No resolution selected...</option>` +
                        data
                );
                adjust_option_width();
            },

            error: function (err) {
                console.log("Ajax error:");
                console.log(err);
            },
        });
    });

    $("#reso-select").on("change", function () {
        $.ajax({
            url: "/resolution/" + this.value,
            type: "POST",
            success: function (data) {
                $("#live-view-container").removeAttr('hidden');
            },
            error: function (err) {
                console.log("Ajax error:");
                console.log(err);
            },
        });
        adjust_option_width();
    });
    
    var save_input = document.getElementById("save-input");
    save_input.addEventListener("keydown", function (e) {
        if (e.code === "Enter") {  //checks whether the pressed key is "Enter"
            update_sub_dirs();
            e.preventDefault();
            return false;
        }
    });


    update_curr_image_dir();
    adjust_option_width();
});

function adjust_option_width() {
    var e = document.querySelectorAll("option");
    e.forEach((x) => {
        if (x.textContent.length > 20)
            x.textContent = x.textContent.substring(0, 20) + "...";
    });
}

function update_sub_dirs() {
    $.ajax({
        url: `/files/directory/list/${encodeURIComponent(encodeURIComponent($("#save-input").val()))}`,
        type: "GET",
        success: function (data) {
            var json_data = $.parseJSON(data);

            container = $("#subdir-buttons");

            content = "";

            for (const dir_name in json_data) {
                content += `<button class="subdir-btn" data-dir-path="${json_data[dir_name]}">${dir_name}</button>` 
            }

            container.html(content);

            $(".subdir-btn").on("pointerup", function() {
                $("#save-input").val($(this).attr("data-dir-path"));
                update_sub_dirs();
            })

        },
        error: function (err) {
            console.log("Ajax error:");
            console.log(err);
        },
    });
}

function update_curr_image_dir() {
    $.ajax({
        url: "/files/directory/get",
        type: "GET",
        success: function (data) {
            $("#save-input").val(data);
            update_sub_dirs();
        },
        error: function (err) {
            console.log("Ajax error:");
            console.log(err);
        },
    });

    
}


window.onresize = adjust_option_width;
