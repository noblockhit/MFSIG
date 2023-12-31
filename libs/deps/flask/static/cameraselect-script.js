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
                $("#reso-select").html(
                    `<option value="-1" selected disabled hidden>No resolution selected...</option>` +
                        data
                );
                adjust_option_width();
            }
        });
    });

    $("#reso-select").on("change", function () {
        $.ajax({
            url: "/resolution/" + this.value,
            type: "POST",
            success: function (data) {
                $("#live-view-container").removeAttr('hidden');
            },
        });
        adjust_option_width();
    });
    
    var save_input = document.getElementById("save-input");
    
    $("#save-form").submit(function (e) {
        document.activeElement.blur()
        update_sub_dirs();
        e.preventDefault();
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

        }
    });
}

function update_curr_image_dir() {
    $.ajax({
        url: "/files/directory/get",
        type: "GET",
        success: function (data) {
            $("#save-input").val(data);
            update_sub_dirs();
        }
    });

    
}


window.onresize = adjust_option_width;
